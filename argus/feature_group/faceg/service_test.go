package faceg

import (
	"context"
	"encoding/json"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"strconv"
	"sync"
	"testing"

	"github.com/qiniu/http/httputil.v1"
	"github.com/stretchr/testify/assert"
	"qbox.us/dht"

	"github.com/qiniu/db/mgoutil.v3"

	FG "qiniu.com/argus/feature_group"
	"qiniu.com/argus/utility/evals"
)

type MockFaceGroupSearch struct {
	Result func(FG.SearchReq) []FG.SearchResult
}

func (s *MockFaceGroupSearch) Search(ctx context.Context, req FG.SearchReq) ([]FG.SearchResult, error) {
	return s.Result(req), nil
}

func (s *MockFaceGroupSearch) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	defer r.Body.Close()
	bs, _ := ioutil.ReadAll(r.Body)
	var req FG.SearchReq
	_ = json.Unmarshal(bs, &req)
	result, _ := s.Search(context.Background(), req)
	rbs, _ := json.Marshal(result)
	w.Write(rbs)
}

type MockFaceDetect struct {
	Result evals.FaceDetectResp
}

func (s *MockFaceDetect) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	rbs, _ := json.Marshal(s.Result)
	w.Write(rbs)
}

type MockFaceFeature struct {
	Result []byte
}

func (s *MockFaceFeature) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	w.Write(s.Result)
}

type MockFaceGroup struct {
	TotalCount  int
	AddFailFace struct {
		FaceId      string
		FailCode    int
		FailMessage string
	}
}

func (s *MockFaceGroup) Hub(ctx context.Context) (hubId FG.HubID, hub FG.Hub)            { return }
func (s *MockFaceGroup) Get(ctx context.Context, id string) (items _FaceItem, err error) { return }
func (s *MockFaceGroup) Del(ctx context.Context, ids []string) (err error)               { return }
func (s *MockFaceGroup) All(ctx context.Context) (items []_FaceItem, err error)          { return }
func (s *MockFaceGroup) Count(ctx context.Context) (count int, err error)                { return }
func (s *MockFaceGroup) Iter(ctx context.Context) (iter FaceGroupIter, err error)        { return }
func (s *MockFaceGroup) CheckByID(ctx context.Context, id string) (res bool, err error)  { return }
func (s *MockFaceGroup) Add(ctx context.Context, items []_FaceItem, features [][]byte) (errs []error) {
	errs = make([]error, s.TotalCount)
	for i, v := range items {
		if v.ID == s.AddFailFace.FaceId {
			errs[i] = httputil.NewError(s.AddFailFace.FailCode, s.AddFailFace.FailMessage)
			break
		}
	}
	return
}

func TestSearch(t *testing.T) {

	hub, _ := FG.NewHubInMgo(
		&mgoutil.Config{DB: "FG_UT"},
		&struct {
			Hubs     mgoutil.Collection `coll:"fg_hub_hubs"`
			Features mgoutil.Collection `coll:"fg_hub_features"`
		}{},
	)
	group, _ := NewFaceGroupManagerInMgo(&mgoutil.Config{DB: "FG_UT"}, hub)

	hub.Clean()
	group.groups.RemoveAll(nil)
	group.faces.RemoveAll(nil)

	search := &MockFaceGroupSearch{Result: nil}
	mux := http.NewServeMux()
	mux.Handle("/v1/search", search)
	ts := httptest.NewServer(mux)
	defer ts.Close()

	nodes := dht.NodeInfos{}
	nodes = append(nodes, dht.NodeInfo{Host: ts.URL})
	service := FaceGroupService{
		_FaceGroupManager: group,
		Interface:         dht.NewCarp(nodes),
		RWMutex:           new(sync.RWMutex),
	}

	ig, _ := group.New(context.Background(), 0x1, "AA", FG.EmptyFeatureVersion)
	for i := 0; i < 99; i++ {
		ig.Add(
			context.Background(),
			[]_FaceItem{{ID: strconv.Itoa(i), Name: strconv.Itoa(i)}},
			[][]byte{make([]byte, 512*4)},
		)
	}
	var v FG.HubVersion = 100

	results := []FG.SearchResult{
		{[]FG.SearchResultItem{{v, 4, 1.0}, {v, 8, 0.8}, {v, 12, 0.6}, {v, 16, 0.4}}},
		{[]FG.SearchResultItem{{v, 60, 0.9}, {v, 59, 0.7}, {v, 58, 0.5}, {v, 57, 0.3}}},
		{[]FG.SearchResultItem{{v, 66, 0.95}, {v, 67, 0.94}, {v, 68, 0.93}, {v, 69, 0.92}}},
		{[]FG.SearchResultItem{{v, 98, 0.955}, {v, 97, 0.45}, {v, 96, 0.43}}},
	}
	var mutex sync.Mutex
	i := 0
	search.Result = func(FG.SearchReq) []FG.SearchResult {
		mutex.Lock()
		defer mutex.Unlock()
		r := results[i]
		i++
		return []FG.SearchResult{r}
	}
	ret, _ := service.postFaceGroupSearch(context.Background(),
		0x1, []string{"AA"}, [][]byte{[]byte{0x1, 0x2}}, []int{512}, []float32{0.525}, 1, 32,
	)
	assert.Equal(t, 1, len(ret))
	assert.Equal(t, "4", ret[0].Name)
}

func TestParseFace(t *testing.T) {
	detect := &MockFaceDetect{Result: evals.FaceDetectResp{
		Result: struct {
			Detections []evals.FaceDetection `json:"detections"`
		}{
			Detections: []evals.FaceDetection{
				evals.FaceDetection{Pts: [][2]int{{10, 80}, {80, 80}, {80, 10}, {80, 10}}},
			},
		},
	}}
	mux_fd := http.NewServeMux()
	mux_fd.Handle("/v1/eval/facex-detect", detect)
	ts_fd := httptest.NewServer(mux_fd)
	defer ts_fd.Close()

	feature := &MockFaceFeature{
		Result: []byte{0, 1, 2, 3, 4, 5},
	}
	mux_ff := http.NewServeMux()
	mux_ff.Handle("/v1/eval/facex-feature", feature)
	ts_ff := httptest.NewServer(mux_ff)
	defer ts_ff.Close()

	s := FaceGroupService{}
	api := NewFaceGFeatureAPI(FaceGFeatureAPIConfig{
		DetectHost:  ts_fd.URL,
		FeatureHost: ts_ff.URL,
	})

	item, ff, _ := s.parseFace(context.Background(), "", "id1234", "name1234", 1, 0, api, "SINGLE")
	assert.Equal(t, "id1234", item.ID)
	assert.Equal(t, "name1234", item.Name)
	assert.Equal(t, []byte{0, 1, 2, 3, 4, 5}, ff)
}

func TestAddFace(t *testing.T) {
	detect := &MockFaceDetect{Result: evals.FaceDetectResp{
		Result: struct {
			Detections []evals.FaceDetection `json:"detections"`
		}{
			Detections: []evals.FaceDetection{
				evals.FaceDetection{Pts: [][2]int{{10, 80}, {80, 80}, {80, 10}, {80, 10}}, Score: 0.88},
			},
		},
	}}
	mux_fd := http.NewServeMux()
	mux_fd.Handle("/v1/eval/facex-detect", detect)
	ts_fd := httptest.NewServer(mux_fd)
	defer ts_fd.Close()

	feature := &MockFaceFeature{
		Result: []byte{0, 1, 2, 3, 4, 5},
	}
	mux_ff := http.NewServeMux()
	mux_ff.Handle("/v1/eval/facex-feature", feature)
	ts_ff := httptest.NewServer(mux_ff)
	defer ts_ff.Close()

	s := FaceGroupService{}
	api := NewFaceGFeatureAPI(FaceGFeatureAPIConfig{
		DetectHost:  ts_fd.URL,
		FeatureHost: ts_ff.URL,
	})

	data := []FaceGroupAddData{
		FaceGroupAddData{
			URI: "uri1",
			Attribute: struct {
				ID   string          `json:"id"`
				Name string          `json:"name"`
				Mode string          `json:"mode"`
				Desc json.RawMessage `json:"desc,omitempty"`
			}{
				ID:   "id1",
				Name: "name1",
			},
		},
		FaceGroupAddData{
			URI: "uri2",
			Attribute: struct {
				ID   string          `json:"id"`
				Name string          `json:"name"`
				Mode string          `json:"mode"`
				Desc json.RawMessage `json:"desc,omitempty"`
			}{
				ID:   "id2",
				Name: "name2",
			},
		},
	}

	//simulate add of second face failed
	fg := &MockFaceGroup{
		TotalCount: 2,
		AddFailFace: struct {
			FaceId      string
			FailCode    int
			FailMessage string
		}{
			FaceId:      "id2",
			FailCode:    400,
			FailMessage: "some error",
		},
	}
	ret := s.addFace(context.Background(), data, fg, api, 1, 0, "group")

	assert.Equal(t, 2, len(ret.Faces))
	assert.Equal(t, 2, len(ret.Attributes))
	assert.Equal(t, 2, len(ret.Errors))

	assert.Equal(t, "id1", ret.Faces[0])
	assert.Equal(t, [][2]int{{10, 80}, {80, 80}, {80, 10}, {80, 10}}, ret.Attributes[0].BoundingBox.Pts)
	assert.Equal(t, float32(0.88), ret.Attributes[0].BoundingBox.Score)
	assert.Nil(t, ret.Errors[0])

	assert.Equal(t, "", ret.Faces[1])
	assert.Nil(t, ret.Attributes[1])
	assert.Equal(t, 400, ret.Errors[1].Code)
	assert.Equal(t, "some error", ret.Errors[1].Message)
}

func Test_GetFaceFeatures(t *testing.T) {
	feature := &MockFaceFeature{
		Result: []byte{0, 1, 2, 3, 4, 5, 6, 7},
	}
	mux_ff := http.NewServeMux()
	mux_ff.Handle("/v1/eval/facex-feature", feature)
	ts_ff := httptest.NewServer(mux_ff)
	defer ts_ff.Close()

	feature3 := &MockFaceFeature{
		Result: []byte{10, 11, 12, 13, 14, 15, 16, 17},
	}
	mux_ff3 := http.NewServeMux()
	mux_ff3.Handle("/v1/eval/facex-feature-v3", feature3)
	ts_ff3 := httptest.NewServer(mux_ff3)
	defer ts_ff3.Close()

	api1 := NewFaceGFeatureAPI(FaceGFeatureAPIConfig{
		FeatureHost: ts_ff.URL,
	})

	api2 := NewFaceGFeatureAPI(FaceGFeatureAPIConfig{
		FeatureHost: ts_ff.URL,
	})

	api3 := NewFaceGFeatureAPI(FaceGFeatureAPIConfig{
		FeatureHost:    ts_ff3.URL,
		FeatureVersion: "-v3",
	})

	s := FaceGroupService{}
	ffs, err := s.getFaceFeatures(context.Background(), []FaceGFeatureAPI{api1, api2, api3}, "", nil, 1, 0)

	assert.Nil(t, err)
	assert.Equal(t, 3, len(ffs))
	assert.Equal(t, []byte{0, 1, 2, 3, 4, 5, 6, 7}, ffs[0])
	assert.Equal(t, []byte{0, 1, 2, 3, 4, 5, 6, 7}, ffs[1])
	assert.Equal(t, []byte{10, 11, 12, 13, 14, 15, 16, 17}, ffs[2])
}
