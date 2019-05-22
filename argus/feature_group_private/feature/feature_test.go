package feature

import (
	"context"
	"encoding/json"
	"io/ioutil"
	"net/http"
	"testing"
	"time"

	"github.com/facebookgo/httpdown"
	restrpc "github.com/qiniu/http/restrpc.v1"
	servestk "github.com/qiniu/http/servestk.v1"
	"github.com/stretchr/testify/assert"
	feature_group "qiniu.com/argus/feature_group_private"
	"qiniu.com/argus/feature_group_private/proto"
)

const (
	mockServerAddr    = "localhost:50001"
	mockGroup         = "test_group"
	mockPatternPrefix = "v1/face"
	imageFeatureSize  = 4096 * 4
	faceFeatureSize   = 512 * 4
)

type mockServer struct {
}

func (s *mockServer) HandleImageFeature(w http.ResponseWriter, r *http.Request) {
	v := make([]byte, imageFeatureSize)
	w.Write(v)
}

func (s *mockServer) HandleFaceFeature(w http.ResponseWriter, r *http.Request) {
	v := make([]byte, faceFeatureSize)
	w.Write(v)
}

func (s *mockServer) HandleFaceDetectV3(w http.ResponseWriter, r *http.Request) {
	type detection struct {
		Index        int                    `json:"index"`
		Class        string                 `json:"class"`
		Score        float32                `json:"score"`
		Pts          [][2]int               `json:"pts"`
		Orientation  proto.FaceOrientation  `json:"orientation,omitempty"`
		Quality      proto.FaceQualityClass `json:"quality,omitempty"`
		QualityScore map[string]float32     `json:"q_score,omitempty"`
	}
	type result struct {
		Detections []detection `json:"detections"`
	}
	type resp struct {
		Code    int    `json:"code"`
		Message string `json:"message"`
		Result  result `json:"result"`
	}
	d := detection{
		Pts: [][2]int{
			[2]int{0, 0},
			[2]int{0, 1000},
			[2]int{1000, 1000},
			[2]int{1000, 0},
		},
	}

	type request struct {
		Data struct {
			URI string `json:"uri"`
		} `json:"data"`
		Params struct {
			UseQuality int `json:"use_quality"`
		} `json:"params"`
	}
	var req request
	body, _ := ioutil.ReadAll(r.Body)
	if err := json.Unmarshal(body, &req); err != nil {
		w.WriteHeader(http.StatusBadRequest)
		return
	}
	if req.Params.UseQuality == 1 {
		d.Quality = proto.FaceQualityClear
		d.Orientation = proto.FaceOrientationUp
		d.QualityScore = map[string]float32{"clear": 0.99999, "blur": 0.00001}
	}

	v, _ := json.Marshal(resp{Result: result{Detections: []detection{d}}})
	w.Write(v)
}

func (s *mockServer) HandleBaseCreateGroup(w http.ResponseWriter, r *http.Request) {
}
func (s *mockServer) HandleBaseRemoveGroup(w http.ResponseWriter, r *http.Request) {
}
func (s *mockServer) HandleBaseAddFeature(w http.ResponseWriter, r *http.Request) {
}
func (s *mockServer) HandleBaseDeleteFeature(w http.ResponseWriter, r *http.Request) {
}
func (s *mockServer) HandleBaseUpdateFeature(w http.ResponseWriter, r *http.Request) {
}
func (s *mockServer) HandleBaseSearchFeature(w http.ResponseWriter, r *http.Request) {
	result := make([][]feature_group.FeatureSearchRespItem, 1)
	result[0] = append(result[0], feature_group.FeatureSearchRespItem{
		Score: 0.5,
		Value: proto.FeatureJson{
			ID: "testid",
		},
	})
	v, _ := json.Marshal(result)
	w.Write(v)
}

func runMockServer() {
	svr := &mockServer{}
	mux := servestk.New(restrpc.DefaultServeMux)
	mux.HandleFunc("POST /v1/eval/image-feature", svr.HandleImageFeature)
	mux.HandleFunc("POST /v1/eval/facex-feature-v4", svr.HandleFaceFeature)
	mux.HandleFunc("POST /v1/eval/facex-detect", svr.HandleFaceDetectV3)
	mux.HandleFunc("POST /"+mockPatternPrefix+"/groups/"+mockGroup, svr.HandleBaseCreateGroup)
	mux.HandleFunc("POST /"+mockPatternPrefix+"/groups/"+mockGroup+"/remove", svr.HandleBaseRemoveGroup)
	mux.HandleFunc("POST /"+mockPatternPrefix+"/groups/"+mockGroup+"/feature/add", svr.HandleBaseAddFeature)
	mux.HandleFunc("POST /"+mockPatternPrefix+"/groups/"+mockGroup+"/delete", svr.HandleBaseDeleteFeature)
	mux.HandleFunc("POST /"+mockPatternPrefix+"/groups/"+mockGroup+"/feature/update", svr.HandleBaseUpdateFeature)
	mux.HandleFunc("POST /"+mockPatternPrefix+"/groups/"+mockGroup+"/feature/search", svr.HandleBaseSearchFeature)
	server := &http.Server{
		Addr:         mockServerAddr,
		Handler:      mux,
		ReadTimeout:  15 * time.Second,
		WriteTimeout: 15 * time.Second,
	}
	hd := &httpdown.HTTP{}
	go func() {
		httpdown.ListenAndServe(server, hd)
	}()
	time.Sleep(time.Second)
}

func TestFeature(t *testing.T) {
	runMockServer()
	a := assert.New(t)
	ctx := context.Background()
	_, err := NewImageFeature("http://"+mockServerAddr, 10*time.Second, imageFeatureSize).Image(ctx, "http://localhost")
	a.Nil(err)
	_, err = NewFaceFeature("http://"+mockServerAddr, 10*time.Second, faceFeatureSize).Face(ctx, "http://localhost", [][2]int{})
	a.Nil(err)
	_, err = NewFaceFeature("http://"+mockServerAddr, 10*time.Second, faceFeatureSize).FaceBoxes(ctx, "http://localhost")
	a.Nil(err)
	_, err = NewFaceFeature("http://"+mockServerAddr, 10*time.Second, faceFeatureSize).FaceBoxesQuality(ctx, "http://localhost")
	a.Nil(err)
	err = NewBaseFeature(10*time.Second, mockPatternPrefix).CreateGroup(ctx, mockServerAddr, mockGroup, proto.GroupConfig{})
	a.Nil(err)
	err = NewBaseFeature(10*time.Second, mockPatternPrefix).RemoveGroup(ctx, mockServerAddr, mockGroup)
	a.Nil(err)
	err = NewBaseFeature(10*time.Second, mockPatternPrefix).AddFeature(ctx, mockServerAddr, mockGroup, proto.Feature{})
	a.Nil(err)
	err = NewBaseFeature(10*time.Second, mockPatternPrefix).DeleteFeature(ctx, mockServerAddr, mockGroup, proto.FeatureID("testid"))
	a.Nil(err)
	err = NewBaseFeature(10*time.Second, mockPatternPrefix).UpdateFeature(ctx, mockServerAddr, mockGroup, proto.Feature{})
	a.Nil(err)
	ret, err := NewBaseFeature(10*time.Second, mockPatternPrefix).SearchFeature(ctx, mockServerAddr, mockGroup, 0, 1, proto.FeatureValue{})
	a.Nil(err)
	a.Equal(ret[0][0].ID, proto.FeatureID("testid"))
}
