package terror_complex

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	"qiniu.com/argus/service/service/image/terror"
	"qiniu.com/argus/utility/evals"
)

func TestTerrorComplex(t *testing.T) {
	var mixup = terror.EvalTerrorMixupEndpoints{EvalTerrorMixupEP: func(ctx context.Context, request interface{}) (interface{}, error) {
		return terror.TerrorMixupResp{
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
						Class: "guns",
						Score: 0.45,
					},
					{
						Index: 40,
						Class: "terrorist",
						Score: 0.55,
					},
					{
						Index: 45,
						Class: "knife",
						Score: 0.99,
					},
				},
				Checkpoint: "endpoint",
			},
		}, nil
	}}

	var detect = terror.EvalTerrorDetectEndpoints{EvalTerrorDetectEP: func(ctx context.Context, request interface{}) (interface{}, error) {
		return evals.TerrorDetectResp{
			Code:    0,
			Message: "",
			Result: struct {
				Detections []evals.TerrorDetection `json:"detections"`
			}{
				Detections: []evals.TerrorDetection{
					{
						Index: 1,
						Class: "knife",
						Score: 0.19,
						Pts:   [][2]int{{225, 195}, {351, 195}, {351, 389}, {225, 389}},
					},
				},
			},
		}, nil
	}}

	var mNormal = terror.EvalTerrorMixupEndpoints{EvalTerrorMixupEP: func(ctx context.Context, request interface{}) (interface{}, error) {
		return terror.TerrorMixupResp{
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
						Index: 1,
						Class: "normal",
						Score: 0.97,
					},
				},
			},
		}, nil
	}}

	var dNormal = terror.EvalTerrorDetectEndpoints{EvalTerrorDetectEP: func(ctx context.Context, request interface{}) (interface{}, error) {
		return evals.TerrorDetectResp{
			Code:    0,
			Message: "",
			Result: struct {
				Detections []evals.TerrorDetection `json:"detections"`
			}{
				Detections: []evals.TerrorDetection{
					{},
				},
			},
		}, nil
	}}

	var mDelay = terror.EvalTerrorMixupEndpoints{EvalTerrorMixupEP: func(ctx context.Context, request interface{}) (interface{}, error) {
		time.Sleep(100 * time.Millisecond)
		return terror.TerrorMixupResp{
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
						Class: "guns",
						Score: 0.97,
					},
				},
			},
		}, nil
	}}

	var dMulti = terror.EvalTerrorDetectEndpoints{EvalTerrorDetectEP: func(ctx context.Context, request interface{}) (interface{}, error) {
		return evals.TerrorDetectResp{
			Code:    0,
			Message: "",
			Result: struct {
				Detections []evals.TerrorDetection `json:"detections"`
			}{
				Detections: []evals.TerrorDetection{
					{
						Index: 1,
						Class: "knife",
						Score: 0.99,
					},
					{
						Index: 1,
						Class: "knife",
						Score: 0.19,
					},
				},
			},
		}, nil
	}}

	//-------------------------------------------------//

	config := Config{TerrorThreshold: 0.25}

	s, _ := NewTerrorComplexService(config, detect, mDelay)
	Resp, err := s.TerrorComplex(context.Background(), terror.TerrorReq{
		Params: struct {
			Detail bool `json:"detail"`
		}{Detail: false}},
	)

	assert.NoError(t, err)
	assert.Equal(t, 0, Resp.Code)
	assert.Equal(t, 1, Resp.Result.Label)
	assert.Equal(t, float32(0.97), Resp.Result.Score)
	assert.Equal(t, false, Resp.Result.Review)

	// s, _ = NewTerrorComplexService(config, k1, k2, nil, k3)
	Resp, err = s.TerrorComplex(context.Background(), terror.TerrorReq{
		Params: struct {
			Detail bool `json:"detail"`
		}{Detail: true}},
	)
	assert.NoError(t, err)
	assert.Equal(t, 0, Resp.Code)
	assert.Equal(t, 1, Resp.Result.Label)
	assert.Equal(t, float32(0.97), Resp.Result.Score)
	assert.Equal(t, float32(0.97), Resp.Result.Classes[0].Score)
	assert.Equal(t, "guns", Resp.Result.Classes[0].Class)
	assert.Equal(t, false, Resp.Result.Review)

	config.TerrorThreshold = 0.15
	s, _ = NewTerrorComplexService(config, detect, mDelay)

	Resp, err = s.TerrorComplex(context.Background(), terror.TerrorReq{
		Params: struct {
			Detail bool `json:"detail"`
		}{Detail: true}},
	)
	assert.NoError(t, err)
	assert.Equal(t, 0, Resp.Code)
	assert.Equal(t, 1, Resp.Result.Label)
	assert.Equal(t, float32(0.97), Resp.Result.Score)
	assert.Equal(t, false, Resp.Result.Review)
	assert.Equal(t, "guns", Resp.Result.Classes[0].Class)
	assert.Equal(t, float32(0.97), Resp.Result.Classes[0].Score)

	s, _ = NewTerrorComplexService(config, detect, mNormal)

	Resp, err = s.TerrorComplex(context.Background(), terror.TerrorReq{
		Params: struct {
			Detail bool `json:"detail"`
		}{Detail: true}},
	)
	assert.NoError(t, err)
	assert.Equal(t, 0, Resp.Code)
	assert.Equal(t, 1, Resp.Result.Label)
	assert.Equal(t, float32(0.19), Resp.Result.Score)
	assert.Equal(t, false, Resp.Result.Review)

	s, _ = NewTerrorComplexService(config, dNormal, mNormal)

	Resp, err = s.TerrorComplex(context.Background(), terror.TerrorReq{
		Params: struct {
			Detail bool `json:"detail"`
		}{Detail: true}},
	)
	assert.NoError(t, err)
	assert.Equal(t, 0, Resp.Code)
	assert.Equal(t, 0, Resp.Result.Label)
	assert.Equal(t, float32(0.97), Resp.Result.Score)
	assert.Equal(t, false, Resp.Result.Review)

	s, _ = NewTerrorComplexService(config, dNormal, mixup)

	Resp, err = s.TerrorComplex(context.Background(), terror.TerrorReq{
		Params: struct {
			Detail bool `json:"detail"`
		}{Detail: true}},
	)
	assert.NoError(t, err)
	assert.Equal(t, 0, Resp.Code)
	assert.Equal(t, 1, Resp.Result.Label)
	assert.Equal(t, float32(0.99), Resp.Result.Score)
	assert.Equal(t, false, Resp.Result.Review)
	assert.Equal(t, "knife", Resp.Result.Classes[0].Class)
	assert.Equal(t, float32(0.99), Resp.Result.Classes[0].Score)

	s, _ = NewTerrorComplexService(config, dMulti, mNormal)

	Resp, err = s.TerrorComplex(context.Background(), terror.TerrorReq{
		Params: struct {
			Detail bool `json:"detail"`
		}{Detail: true}},
	)
	assert.NoError(t, err)
	assert.Equal(t, 0, Resp.Code)
	assert.Equal(t, 1, Resp.Result.Label)
	assert.Equal(t, float32(0.99), Resp.Result.Score)
	assert.Equal(t, false, Resp.Result.Review)
	assert.Equal(t, "knife", Resp.Result.Classes[0].Class)
	assert.Equal(t, float32(0.99), Resp.Result.Classes[0].Score)

}
