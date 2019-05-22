package imageg

import (
	"context"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"strconv"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"qbox.us/dht"
	httputil "qiniupkg.com/http/httputil.v2"

	"github.com/qiniu/db/mgoutil.v3"

	URI "qiniu.com/argus/argus/com/uri"
	FG "qiniu.com/argus/feature_group"
)

type MockImageGroupSearch struct {
	Result func(FG.SearchReq) []FG.SearchResult
}

func (s *MockImageGroupSearch) Search(ctx context.Context, req FG.SearchReq) ([]FG.SearchResult, error) {
	return s.Result(req), nil
}

func (s *MockImageGroupSearch) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	defer r.Body.Close()
	bs, _ := ioutil.ReadAll(r.Body)
	var req FG.SearchReq
	_ = json.Unmarshal(bs, &req)
	result, _ := s.Search(context.Background(), req)
	rbs, _ := json.Marshal(result)
	w.Write(rbs)
}

type MockImageFeature struct {
	Result []byte
}

func (s *MockImageFeature) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	w.Write(s.Result)
}

type MockImageGroup struct {
	TotalCount   int
	AddFailImage struct {
		ImageId     string
		FailCode    int
		FailMessage string
	}
}

func (s *MockImageGroup) Hub(ctx context.Context) (hubId FG.HubID, hub FG.Hub)             { return }
func (s *MockImageGroup) Get(ctx context.Context, id string) (items _ImageItem, err error) { return }
func (s *MockImageGroup) Del(ctx context.Context, ids []string) (err error)                { return }
func (s *MockImageGroup) All(ctx context.Context) (items []_ImageItem, err error)          { return }
func (s *MockImageGroup) Count(ctx context.Context) (count int, err error)                 { return }
func (s *MockImageGroup) Iter(ctx context.Context) (iter ImageGroupIter, err error)        { return }
func (s *MockImageGroup) CheckByID(ctx context.Context, id string) (res bool, err error)   { return }
func (s *MockImageGroup) Add(ctx context.Context, items []_ImageItem, features [][]byte) (errs []error) {
	errs = make([]error, s.TotalCount)
	for i, v := range items {
		if v.ID == s.AddFailImage.ImageId {
			errs[i] = httputil.NewError(s.AddFailImage.FailCode, s.AddFailImage.FailMessage)
			break
		}
	}
	return
}

func TestSearch(t *testing.T) {

	hub, _ := FG.NewHubInMgo(
		&mgoutil.Config{DB: "IG_UT_SERVICE"},
		&struct {
			Hubs     mgoutil.Collection `coll:"ig_hub_hubs"`
			Features mgoutil.Collection `coll:"ig_hub_features"`
		}{},
	)
	group, _ := NewImageGroupManagerInMgo(&mgoutil.Config{DB: "IG_UT"}, hub)

	hub.Clean()
	group.groups.RemoveAll(nil)
	group.images.RemoveAll(nil)

	search := &MockImageGroupSearch{Result: nil}
	mux := http.NewServeMux()
	mux.Handle("/v1/search", search)
	ts := httptest.NewServer(mux)
	defer ts.Close()

	nodes := dht.NodeInfos{}
	nodes = append(nodes, dht.NodeInfo{Host: ts.URL})
	service := ImageGroupService{
		_ImageGroupManager: group,
		Interface:          dht.NewCarp(nodes),
		RWMutex:            new(sync.RWMutex),
	}

	ig, _ := group.New(context.Background(), 0x1, "AA", FG.EmptyFeatureVersion)
	for i := 0; i < 99; i++ {
		ig.Add(
			context.Background(),
			[]_ImageItem{{ID: strconv.Itoa(i), Label: strconv.Itoa(i)}},
			[][]byte{make([]byte, 4*1024*4)},
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
	ret, _ := service.postImageGroupSearch(context.Background(),
		0x1, []string{"AA"}, [][]byte{[]byte{0x1, 0x2}}, []int{4096}, []float32{0}, 4, 32,
	)
	assert.Equal(t, 4, len(ret))
	assert.Equal(t, "4", ret[0].Label)
	assert.Equal(t, "98", ret[1].Label)
	assert.Equal(t, "66", ret[2].Label)
	assert.Equal(t, "67", ret[3].Label)
}

func TestNewImageGFeatureAPI(t *testing.T) {
	api := NewImageGFeatureAPI(ImageGFeatureAPIConfig{})
	assert.NotNil(t, api)
}

func TestNewImageGService(t *testing.T) {
	s, sFunc := NewImageGroupServic(Config{}, nil, nil, nil)
	assert.NotNil(t, s)
	assert.NotNil(t, sFunc)
}

func TestEtag(t *testing.T) {
	origin := "test1234"
	res := []byte{155, 195, 69, 73, 213, 101, 217, 80, 91, 40, 125, 224, 205, 32, 172, 119, 190, 29, 63, 44}
	str := base64.StdEncoding.EncodeToString([]byte(origin))
	etag, _ := dataURIEtag(URI.DataURIPrefix + str)
	assert.Equal(t, hex.Dump(res), etag)
}

func TestAddImage(t *testing.T) {
	feature := &MockImageFeature{
		Result: []byte{0, 1, 2, 3, 4, 5, 6, 7},
	}
	mux_ff := http.NewServeMux()
	mux_ff.Handle("/v1/eval/image-feature", feature)
	ts_ff := httptest.NewServer(mux_ff)
	defer ts_ff.Close()

	s := ImageGroupService{}
	api := NewImageGFeatureAPI(ImageGFeatureAPIConfig{
		FeatureHost: ts_ff.URL,
	})

	data := []ImageGroupAddData{
		ImageGroupAddData{
			URI: "uri1",
			Attribute: struct {
				ID    string          `json:"id"`
				Label string          `json:"label"`
				Desc  json.RawMessage `json:"desc,omitempty"`
			}{
				ID:    "id1",
				Label: "label1",
			},
		},
		ImageGroupAddData{
			URI: "uri2",
			Attribute: struct {
				ID    string          `json:"id"`
				Label string          `json:"label"`
				Desc  json.RawMessage `json:"desc,omitempty"`
			}{
				ID:    "id2",
				Label: "label2",
			},
		},
	}

	//simulate add of second face failed
	fg := &MockImageGroup{
		TotalCount: 2,
		AddFailImage: struct {
			ImageId     string
			FailCode    int
			FailMessage string
		}{
			ImageId:     "id2",
			FailCode:    400,
			FailMessage: "some error",
		},
	}
	ret := s.addImage(context.Background(), data, fg, api, 1, 0, "group")

	assert.Equal(t, 2, len(ret.Images))
	assert.Equal(t, 2, len(ret.Errors))

	assert.Equal(t, "id1", ret.Images[0])
	assert.Nil(t, ret.Errors[0])

	assert.Equal(t, "", ret.Images[1])
	assert.Equal(t, 400, ret.Errors[1].Code)
	assert.Equal(t, "some error", ret.Errors[1].Message)
}

func Test_GetImageFeatures(t *testing.T) {
	feature := &MockImageFeature{
		Result: []byte{0, 1, 2, 3, 4, 5, 6, 7},
	}
	mux_ff := http.NewServeMux()
	mux_ff.Handle("/v1/eval/image-feature", feature)
	ts_ff := httptest.NewServer(mux_ff)
	defer ts_ff.Close()

	api1 := NewImageGFeatureAPI(ImageGFeatureAPIConfig{
		FeatureHost: ts_ff.URL,
	})

	s := ImageGroupService{}
	ffs, err := s.getImageFeatures(context.Background(), []ImageGFeatureAPI{api1}, "", 1, 0)

	assert.Nil(t, err)
	assert.Equal(t, 1, len(ffs))
	assert.Equal(t, []byte{0, 0, 128, 127, 0, 0, 128, 127}, ffs[0])
}
