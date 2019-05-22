package politician

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"qiniu.com/argus/utility/evals"
)

var efds = EvalFaceDetectEndpoints{EvalFaceDetectEP: func(ctx context.Context, request interface{}) (interface{}, error) {
	resp := evals.FaceDetectResp{}
	resp.Result.Detections = []evals.FaceDetection{
		evals.FaceDetection{
			Index: 1,
			Class: "face",
			Score: 0.9971,
			Pts:   [][2]int{{225, 195}, {351, 195}, {351, 389}, {225, 389}},
		},
	}
	return resp, nil
}}

var effs = EvalFaceFeatureEndpoints{EvalFaceFeatureEP: func(ctx context.Context, request interface{}) (interface{}, error) {
	return []byte("feature string"), nil
}}

var eps = EvalPoliticianEndpoints{EvalPoliticianEP: func(ctx context.Context, request interface{}) (interface{}, error) {
	var resp evals.FaceSearchRespV2
	resp.Result.Confidences = append(
		resp.Result.Confidences, struct {
			Index  int     `json:"index"`
			Class  string  `json:"class"`
			Group  string  `json:"group"`
			Score  float32 `json:"score"`
			Sample struct {
				URL string   `json:"url"`
				Pts [][2]int `json:"pts"`
				ID  string   `json:"id"`
			} `json:"sample"`
		}{
			Class: "XXX",
			Group: "affairs_official_gov",
			Score: 0.998,
		},
	)
	return resp, nil
}}

func TestPolitician(t *testing.T) {

	config := Config{PoliticianThreshold: []float32{0.6, 0.66, 0.72}}
	s, _ := NewFaceSearchService(config, efds, effs, eps)
	Resp, err := s.FaceSearch(context.Background(), Req{})

	assert.NoError(t, err)
	assert.Equal(t, 0, Resp.Code)
	assert.Equal(t, float32(0.9971), Resp.Result.Detections[0].BoundingBox.Score)
	assert.Equal(t, "XXX", Resp.Result.Detections[0].Value.Name)
	assert.Equal(t, "affairs_official_gov", Resp.Result.Detections[0].Value.Group)
	assert.Equal(t, float32(0.998), Resp.Result.Detections[0].Value.Score)

}
