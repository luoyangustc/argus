package pulp

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"qiniu.com/argus/utility/evals"
)

var epps = EvalPulpEndpoints{EvalPulpEP: func(ctx context.Context, request interface{}) (interface{}, error) {
	return evals.PulpResp{
		Code:    0,
		Message: "",
		Result: struct {
			Checkpoint  string
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
					Class: "sexy",
					Index: 1,
					Score: 0.9,
				},
				{
					Class: "normal",
					Index: 2,
					Score: 0.2,
				},
				{
					Class: "pulp",
					Index: 0,
					Score: 0.1,
				},
			},
		},
	}, nil
}}

var epps2 = EvalPulpEndpoints{EvalPulpEP: func(ctx context.Context, request interface{}) (interface{}, error) {
	return evals.PulpResp{
		Code:    0,
		Message: "",
		Result: struct {
			Checkpoint  string
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
					Class: "sexy",
					Index: 1,
					Score: 0.3,
				},
				{
					Class: "normal",
					Index: 2,
					Score: 0.2,
				},
				{
					Class: "pulp",
					Index: 0,
					Score: 0.9,
				},
			},
		},
	}, nil
}}

var epss1 = EvalPulpFilterEndpoints{EvalPulpFilterEP: func(ctx context.Context, request interface{}) (interface{}, error) {
	return evals.PulpResp{
		Code:    0,
		Message: "",
		Result: struct {
			Checkpoint  string
			Confidences []struct {
				Index int     `json:"index"`
				Class string  `json:"class"`
				Score float32 `json:"score"`
			} `json:"confidences"`
		}{
			Checkpoint: "endpoint",
			Confidences: []struct {
				Index int     `json:"index"`
				Class string  `json:"class"`
				Score float32 `json:"score"`
			}{
				{
					Class: "normal",
					Index: 2,
					Score: 0.59,
				},
				{
					Class: "sexy",
					Index: 1,
					Score: 0.5,
				},
				{
					Class: "pulp",
					Index: 0,
					Score: 0.4,
				},
			},
		},
	}, nil
}}

var epss2 = EvalPulpFilterEndpoints{EvalPulpFilterEP: func(ctx context.Context, request interface{}) (interface{}, error) {
	return evals.PulpResp{
		Code:    0,
		Message: "",
		Result: struct {
			Checkpoint  string
			Confidences []struct {
				Index int     `json:"index"`
				Class string  `json:"class"`
				Score float32 `json:"score"`
			} `json:"confidences"`
		}{
			Checkpoint: "XXX",
			Confidences: []struct {
				Index int     `json:"index"`
				Class string  `json:"class"`
				Score float32 `json:"score"`
			}{
				{
					Class: "sexy",
					Index: 2,
					Score: 0.61,
				},
				{
					Class: "normal",
					Index: 1,
					Score: 0.5,
				},
				{
					Class: "pulp",
					Index: 0,
					Score: 0.4,
				},
			},
		},
	}, nil
}}

func TestPulp(t *testing.T) {

	config := Config{PulpReviewThreshold: 0.6}
	s1, _ := NewPulpService(config, epps, epss1)
	pResp1, err := s1.Pulp(context.Background(), PulpReq{
		Params: struct {
			Limit int `json:"limit"`
		}{
			Limit: 3,
		}})

	assert.NoError(t, err)
	assert.Equal(t, 0, pResp1.Code)
	assert.Equal(t, "", pResp1.Message)
	assert.Equal(t, 2, pResp1.Result.Label)
	assert.Equal(t, float32(0.59), pResp1.Result.Score)
	assert.Equal(t, true, pResp1.Result.Review)
	assert.Equal(t, "normal", pResp1.Result.Confidences[0].Class)
	assert.Equal(t, 2, pResp1.Result.Confidences[0].Index)
	assert.Equal(t, float32(0.59), pResp1.Result.Confidences[0].Score)

	s2, _ := NewPulpService(config, epps, epss2)
	pResp2, err := s2.Pulp(context.Background(), PulpReq{
		Params: struct {
			Limit int `json:"limit"`
		}{
			Limit: 3,
		}})

	assert.NoError(t, err)
	assert.Equal(t, 0, pResp2.Code)
	assert.Equal(t, 1, pResp2.Result.Label)
	assert.Equal(t, float32(0.9), pResp2.Result.Score)
	assert.Equal(t, false, pResp2.Result.Review)
}
