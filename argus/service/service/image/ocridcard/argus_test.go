package ocridcard

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	pimage "qiniu.com/argus/service/service/image"
	simage "qiniu.com/argus/service/service/image"
)

var eops = EvalOcrSariIdcardPreProcessEndpoints{EvalOcrSariIdcardPreProcessEP: func(ctx context.Context, request interface{}) (interface{}, error) {
	req := request.(EvalOcrSariIdcardPreProcessReq)
	if req.Params.Type == "predetect" {
		// return resp, nil
		return EvalOcrSariIdcardPreProcessResp{
			Code:    0,
			Message: "",
			Result: struct {
				Class         int               `json:"class"`
				AlignedImg    string            `json:"alignedImg"`
				Names         []string          `json:"names"`
				Regions       [][4][2]int       `json:"regions"`
				Bboxes        [][4][2]int       `json:"bboxes"`
				DetectedBoxes [][8]int          `json:"detectedBoxes"`
				Res           map[string]string `json:"res"`
			}{
				AlignedImg: "http://p9zv90cqq.bkt.clouddn.com/alignedimg/cut_001.jpg",
				Bboxes: [][4][2]int{
					{{120, 225}, {120, 270}, {440, 270}, {440, 225}},
					{{120, 305}, {120, 350}, {440, 350}, {440, 305}},
					{{120, 265}, {120, 310}, {440, 310}, {440, 265}},
					{{35, 115}, {35, 155}, {365, 155}, {365, 115}},
					{{225, 365}, {225, 415}, {690, 415}, {690, 365}},
					{{120, 165}, {120, 210}, {370, 210}, {370, 165}},
					{{135, 50}, {135, 100}, {212, 100}, {212, 50}},
				},
				Class: 0,
				Names: []string{"住址1", "住址3", "住址2", "性民", "公民身份号码", "出生", "姓名"},
				Regions: [][4][2]int{
					{{120, 225}, {120, 270}, {440, 270}, {440, 225}},
					{{120, 305}, {120, 350}, {440, 350}, {440, 305}},
					{{120, 265}, {120, 310}, {440, 310}, {440, 265}},
					{{35, 115}, {35, 155}, {365, 155}, {365, 115}},
					{{225, 365}, {225, 415}, {690, 415}, {690, 365}},
					{{120, 165}, {120, 210}, {370, 210}, {370, 165}},
					{{135, 50}, {135, 100}, {212, 100}, {212, 50}},
				},
			},
		}, nil
	} else if req.Params.Type == "prerecog" {
		return EvalOcrSariIdcardPreProcessResp{
			Code:    0,
			Message: "",
			Result: struct {
				Class         int               `json:"class"`
				AlignedImg    string            `json:"alignedImg"`
				Names         []string          `json:"names"`
				Regions       [][4][2]int       `json:"regions"`
				Bboxes        [][4][2]int       `json:"bboxes"`
				DetectedBoxes [][8]int          `json:"detectedBoxes"`
				Res           map[string]string `json:"res"`
			}{
				Bboxes: [][4][2]int{
					{{134, 227}, {419, 227}, {419, 262}, {134, 262}},
					{{134, 268}, {235, 268}, {235, 296}, {134, 296}},
					{{35, 115}, {35, 155}, {365, 155}, {365, 115}},
					{{634, 369}, {635, 402}, {226, 406}, {225, 373}},
					{{120, 165}, {120, 210}, {370, 210}, {370, 165}},
					{{115, 50}, {115, 100}, {232, 100}, {232, 50}},
				},
			},
		}, nil
	} else {
		return EvalOcrSariIdcardPreProcessResp{
			Code:    0,
			Message: "",
			Result: struct {
				Class         int               `json:"class"`
				AlignedImg    string            `json:"alignedImg"`
				Names         []string          `json:"names"`
				Regions       [][4][2]int       `json:"regions"`
				Bboxes        [][4][2]int       `json:"bboxes"`
				DetectedBoxes [][8]int          `json:"detectedBoxes"`
				Res           map[string]string `json:"res"`
			}{
				Res: map[string]string{
					"住址":     "河南省项城市芙蓉巷东四胡同2号",
					"公民身份号码": "412702199705127504",
					"出生":     "1997年5月12日",
					"姓名":     "张杰",
					"性别":     "女",
					"民族":     "汉",
				},
			},
		}, nil
	}
},
}

var eods = EvalOcrSariIdcardDetectEndpoints{EvalOcrSariIdcardDetectEP: func(ctx context.Context, request interface{}) (interface{}, error) {
	return EvalOcrSariIdcardDetectResp{
		Code:    0,
		Message: "",
		Result: struct {
			Bboxes [][8]int `json:"bboxes"`
		}{
			Bboxes: [][8]int{
				{121, 231, 413, 229, 413, 260, 121, 263},
				{72, 173, 346, 176, 345, 205, 72, 202},
				{45, 62, 215, 60, 215, 93, 46, 95},
				{243, 374, 619, 371, 620, 404, 244, 407},
				{131, 270, 239, 270, 239, 299, 131, 298},
				{46, 374, 205, 373, 206, 403, 46, 404},
				{47, 122, 162, 120, 162, 150, 47, 151},
				{206, 121, 302, 121, 302, 148, 206, 148},
				{44, 232, 119, 233, 118, 259, 44, 258},
			},
		},
	}, nil
},
}

var eors = EvalOcrSariIdcardRecogEndpoints{EvalOcrSariIdcardRecogEP: func(ctx context.Context, request interface{}) (interface{}, error) {
	return EvalOcrSariIdcardRecogResp{
		Code:    0,
		Message: "",
		Result: struct {
			Texts []string `json:"text"`
		}{
			Texts: []string{
				"河南省项城市芙蓉巷东四",
				"胡同2号",
				"性别‘女一民族汉",
				"412702199705127504",
				"1997年5月12日",
				"张杰",
			},
		},
	}, nil
},
}

func TestOcrSariIdcard(t *testing.T) {
	s, _ := NewOcrIdcardService(eods, eors, eops)
	resp, err := s.OcrIdcard(context.Background(), OcrSariIdcardReq{
		Data: struct {
			IMG pimage.Image
			// URI string `json:"uri"`
		}{
			IMG: struct {
				Format string        `json:"format"`
				Width  int           `json:"width"`
				Height int           `json:"height"`
				URI    simage.STRING `json:"uri"`
			}{
				URI: simage.STRING("http://p9zv90cqq.bkt.clouddn.com/alignedimg/cut_001.jpg"),
			},
		},
	})

	assert.Nil(t, err)
	assert.Equal(t, 0, resp.Result.Type)
	assert.Equal(t, 0, resp.Code)
	assert.Equal(t, "http://p9zv90cqq.bkt.clouddn.com/alignedimg/cut_001.jpg", resp.Result.URI)
	for k, v := range resp.Result.Res {
		if k == "住址" {
			assert.Equal(t, "河南省项城市芙蓉巷东四胡同2号", v)
		} else if k == "公民身份号码" {
			assert.Equal(t, "412702199705127504", v)
		} else if k == "出生" {
			assert.Equal(t, "1997年5月12日", v)
		} else if k == "姓名" {
			assert.Equal(t, "张杰", v)
		} else if k == "性别" {
			assert.Equal(t, "女", v)
		} else {
			assert.Equal(t, "汉", v)
		}
	}
}
