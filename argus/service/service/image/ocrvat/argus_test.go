package ocrvat

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	pimage "qiniu.com/argus/service/service/image"
)

var vds = EvalOcrSariVatDetectEndpoints{EvalOcrSariVatDetectEP: func(ctx context.Context, request interface{}) (interface{}, error) {
	return EvalOcrSariVatDetectResp{
		Code:    0,
		Message: "",
		Result: struct {
			Bboxes [][4][2]float32 `json:"bboxes"`
		}{
			Bboxes: [][4][2]float32{{{1, 1}, {2, 2}, {3, 3}, {4, 4}}, {{1, 1}, {2, 2}, {3, 3}, {4, 4}}},
		},
	}, nil
}}

var vrs = EvalOcrSariVatRecogEndpoints{EvalOcrSariVatRecogEP: func(ctx context.Context, request interface{}) (interface{}, error) {
	return EvalOcrSariVatRecogResp{
		Code:    0,
		Message: "",
		Result: struct {
			Texts []string `json:"text"`
		}{
			Texts: []string{"hello", "world"},
		},
	}, nil
}}

var vpps = EvalOcrSariVatPostProcessEndpoints{EvalOcrSariVatPostProcessEP: func(ctx context.Context, request interface{}) (interface{}, error) {
	return EvalOcrSariVatPostProcessResp{
		Code:    0,
		Message: "",
		Result: map[string]interface{}{
			"_XiaoLeiMingCheng":           "一般增值税发票",
			"_XiaoShouFangDiZhiJiDianHua": "成都市高新区高新大道创业路14-16号 028-87492600",
		},
	}, nil
}}

func TestOcrSariVat(t *testing.T) {
	s, _ := NewOcrSariVatService(vds, vrs, vpps)
	resp, err := s.OcrSariVat(context.Background(), OcrSariVatReq{
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

	var bboxes = [][4][2]float32{{{1, 1}, {2, 2}, {3, 3}, {4, 4}}, {{1, 1}, {2, 2}, {3, 3}, {4, 4}}}
	var texts = map[string]interface{}{
		"_XiaoLeiMingCheng":           "一般增值税发票",
		"_XiaoShouFangDiZhiJiDianHua": "成都市高新区高新大道创业路14-16号 028-87492600",
	}
	assert.Nil(t, err)
	assert.Equal(t, 0, resp.Code)
	assert.Equal(t, bboxes, resp.Result.Bboxes)
	assert.Equal(t, texts, resp.Result.Res)
}
