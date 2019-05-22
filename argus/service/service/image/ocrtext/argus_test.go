package ocrtext

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	pimage "qiniu.com/argus/service/service/image"
)

var eocs = EvalOcrTextClassifyEndpoints{EvalOcrTextClassifyEP: func(ctx context.Context, request interface{}) (interface{}, error) {
	return EvalOcrTextClassifyResp{
		Code:    0,
		Message: "",
		Result: struct {
			Confidences []struct {
				Index int     `json:"index"`
				Class string  `json:"class"`
				Score float64 `json:"score"`
			} `json:"confidences"`
		}{
			Confidences: []struct {
				Index int     `json:"index"`
				Class string  `json:"class"`
				Score float64 `json:"score"`
			}{
				{
					Index: 31,
					Class: "weibo",
					Score: 0.999,
				},
			},
		},
	}, nil
},
}

var eods = EvalOcrCtpnEndpoints{EvalOcrCtpnEP: func(ctx context.Context, request interface{}) (interface{}, error) {
	return EvalOcrCtpnResp{
		Code:    0,
		Message: "",
		Result: struct {
			Bboxes [][4][2]int `json:"bboxes"`
		}{
			Bboxes: [][4][2]int{
				{{10, 20}, {50, 20}, {50, 100}, {10, 100}},
				{{60, 43}, {70, 43}, {70, 120}, {60, 120}},
			},
		},
	}, nil
},
}

var eors = EvalOcrTextRecognizeEndpoints{EvalOcrTextRecognizeEP: func(ctx context.Context, request interface{}) (interface{}, error) {
	return EvalOcrTextRecognizeResp{
		Code:    0,
		Message: "",
		Result: struct {
			Bboxes [][4][2]int `json:"bboxes"`
			Texts  []string    `json:"texts"`
		}{
			Bboxes: [][4][2]int{
				{{10, 20}, {50, 20}, {50, 100}, {10, 100}},
				{{60, 43}, {70, 43}, {70, 120}, {60, 120}},
			},
			Texts: []string{
				"this mock ocr text test",
				"this mock ocr text test, again",
			},
		},
	}, nil
}}

var eosds = EvalOcrSceneDetectEndpoints{EvalOcrSceneDetectEP: func(ctx context.Context, request interface{}) (interface{}, error) {
	return EvalOcrSceneDetectResp{
		Code:    0,
		Message: "",
		Result: struct {
			Bboxes [][8]int `json:"bboxes"`
		}{
			Bboxes: [][8]int{
				{10, 20, 50, 100, 60, 43, 70, 120},
			},
		},
	}, nil
}}

var eosrs = EvalOcrSceneRecognizeEndpoints{EvalOcrSceneRecognizeEP: func(ctx context.Context, request interface{}) (interface{}, error) {
	return EvalOcrSceneRecognizeResp{
		Code:    0,
		Message: "",
		Result: struct {
			Texts []OcrSceneRespResult `json:"texts"`
		}{
			Texts: []OcrSceneRespResult{
				OcrSceneRespResult{
					Bboxes: [8]int{10, 20, 50, 100, 60, 43, 70, 120},
					Text:   "this mock ocr text test",
				},
			},
		},
	}, nil
}}

func TestOcrText(t *testing.T) {
	s, _ := NewOcrTextService(eocs, eods, eors, eosds, eosrs)
	resp, err := s.OcrText(context.Background(), OcrTextReq{
		Data: struct {
			IMG pimage.Image
		}{
			IMG: struct {
				Format string        `json:"format"`
				Width  int           `json:"width"`
				Height int           `json:"height"`
				URI    pimage.STRING `json:"uri"`
			}{
				URI: pimage.STRING("http://test.image.jpg"),
			},
		},
	})

	text := []string{"this mock ocr text test", "this mock ocr text test, again"}
	bboxes := [][4][2]int{{{10, 20}, {50, 20}, {50, 100}, {10, 100}}, {{60, 43}, {70, 43}, {70, 120}, {60, 120}}}
	assert.Nil(t, err)
	assert.Equal(t, 0, resp.Code)
	assert.Equal(t, "weibo", resp.Result.Type)
	assert.Equal(t, bboxes, resp.Result.Bboxes)
	assert.Equal(t, text, resp.Result.Texts)
}
