package ocrbankcard

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	pimage "qiniu.com/argus/service/service/image"
	simage "qiniu.com/argus/service/service/image"
)

var eods = EvalOcrSariBankcardDetectEndpoints{EvalOcrSariBankcardDetectEP: func(ctx context.Context, request interface{}) (interface{}, error) {
	return EvalOcrSariBankcardDetectResp{
		Code:    0,
		Message: "",
		Result: struct {
			Bboxes [][4][2]int `json:"bboxes"`
		}{
			Bboxes: [][4][2]int{
				{{850, 91}, {965, 87}, {968, 162}, {853, 166}},
				{{149, 97}, {757, 82}, {759, 159}, {151, 175}},
				{{284, 178}, {738, 181}, {738, 205}, {284, 201}},
				{{853, 183}, {966, 184}, {965, 213}, {852, 212}},
				{{208, 270}, {984, 267}, {985, 373}, {208, 376}},
				{{132, 378}, {947, 385}, {946, 450}, {132, 442}},
				{{132, 492}, {207, 491}, {207, 533}, {132, 534}},
				{{454, 492}, {586, 491}, {586, 538}, {455, 538}},
				{{644, 496}, {738, 496}, {738, 541}, {644, 540}},
				{{834, 515}, {966, 517}, {965, 586}, {833, 584}},
			},
		},
	}, nil
},
}

var eors = EvalOcrSariBankcardRecogEndpoints{EvalOcrSariBankcardRecogEP: func(ctx context.Context, request interface{}) (interface{}, error) {
	return EvalOcrSariBankcardRecogResp{
		Code:    0,
		Message: "",
		Result: struct {
			Texts []string `json:"text"`
		}{
			Texts: []string{
				"9558801001128579882",
				"邑)中国工商银行",
			},
		},
	}, nil
}}

func TestOcrSariBankcard(t *testing.T) {
	s, _ := NewOcrBankcardService(eods, eors)
	resp, err := s.OcrBankcard(context.Background(), OcrSariBankcardReq{
		Data: struct {
			IMG pimage.Image
		}{
			IMG: struct {
				Format string        `json:"format"`
				Width  int           `json:"width"`
				Height int           `json:"height"`
				URI    simage.STRING `json:"uri"`
			}{
				URI: simage.STRING("http://pfnm7b5zm.bkt.clouddn.com/7.jpg"),
			},
		},
	})

	assert.Nil(t, err)
	assert.Equal(t, 0, resp.Code)
	assert.Equal(t, "", resp.Message)

	for k, v := range resp.Result.Res {
		if k == "开户银行" {
			assert.Equal(t, "邑)中国工商银行", v)
		} else {
			assert.Equal(t, "9558801001128579882", v)
		}
	}

	bboxes := [][4][2]int{
		{{850, 91}, {965, 87}, {968, 162}, {853, 166}},
		{{149, 97}, {757, 82}, {759, 159}, {151, 175}},
	}

	assert.Equal(t, bboxes, resp.Result.Bboxes)
}
