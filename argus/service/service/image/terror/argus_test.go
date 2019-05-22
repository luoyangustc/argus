package terror

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"qiniu.com/argus/utility/evals"
)

var etpds = EvalTerrorMixupEndpoints{EvalTerrorMixupEP: func(ctx context.Context, request interface{}) (interface{}, error) {
	return TerrorMixupResp{
		Code:    0,
		Message: "",
		Result: struct {
			Checkpoint  string `json:"checkpoint"`
			Confidences []struct {
				Index int     `json:"index"`
				Class string  `json:"class"`
				Score float32 `json:"score"`
			} `json:"confidences"`
		}{
			Confidences: []struct {
				Index int     `json:"index"`
				Class string  `json:"class"`
				Score float32 `json:"score"`
			}{
				{
					Index: 47,
					Class: "bloodiness",
					Score: 0.45,
				},
				{
					Index: 40,
					Class: "fight_police",
					Score: 0.55,
				},
				{
					Index: 45,
					Class: "beheaded",
					Score: 0.69,
				},
			},
			Checkpoint: "terror-detect",
		},
	}, nil
}}

var etds = EvalTerrorDetectEndpoints{EvalTerrorDetectEP: func(ctx context.Context, request interface{}) (interface{}, error) {
	return evals.TerrorDetectResp{
		Code:    0,
		Message: "",
		Result: struct {
			Detections []evals.TerrorDetection `json:"detections"`
		}{
			Detections: []evals.TerrorDetection{
				{
					Index: 1,
					Class: "knives",
					Score: 0.79,
					Pts:   [][2]int{{225, 195}, {351, 195}, {351, 389}, {225, 389}},
				},
				{
					Index: 2,
					Class: "guns",
					Score: 0.5,
					Pts:   [][2]int{{225, 195}, {351, 195}, {351, 389}, {225, 389}},
				},
				{
					Index: 3,
					Class: "guns",
					Score: 0.1,
					Pts:   [][2]int{{225, 195}, {351, 195}, {351, 389}, {225, 389}},
				},
				{
					Index: 1,
					Class: "knives",
					Score: 0.69,
					Pts:   [][2]int{{225, 195}, {351, 195}, {351, 389}, {225, 389}},
				},
			},
		},
	}, nil
}}

func TestTerror(t *testing.T) {

	config := Config{TerrorThreshold: 0.85}
	s, _ := NewTerrorService(config, etpds, etds)
	Resp, err := s.Terror(context.Background(),
		TerrorReq{
			Params: struct {
				Detail bool `json:"detail"`
			}{Detail: true},
		})

	assert.NoError(t, err)
	assert.Equal(t, 0, Resp.Code)
	assert.Equal(t, 1, Resp.Result.Label)
	assert.Equal(t, float32(0.79), Resp.Result.Score)
	assert.Equal(t, true, Resp.Result.Review)
	assert.Equal(t, "knives", Resp.Result.Class)
}
