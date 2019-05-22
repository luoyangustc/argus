package ocrvat

import (
	"context"
	"github.com/go-kit/kit/endpoint"
	pimage "qiniu.com/argus/service/service/image"
)

//-------------------------------------------------------------------------------
//文字检测
type EvalOcrSariVatDetectReq struct {
	Data struct {
		IMG pimage.Image
	} `json:"data"`
	Params struct {
		Type string `json:"type"`
	} `json:"params"`
}

type EvalOcrSariVatDetectResp struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Result  struct {
		Bboxes [][4][2]float32 `json:"bboxes"`
	} `json:"result"`
}

type EvalOcrSariVatDetectService interface {
	EvalOcrSariVatDetect(ctx context.Context, img EvalOcrSariVatDetectReq) (EvalOcrSariVatDetectResp, error)
}

var _ EvalOcrSariVatDetectService = EvalOcrSariVatDetectEndpoints{}

type EvalOcrSariVatDetectEndpoints struct {
	EvalOcrSariVatDetectEP endpoint.Endpoint
}

func (ends EvalOcrSariVatDetectEndpoints) EvalOcrSariVatDetect(ctx context.Context, img EvalOcrSariVatDetectReq) (
	EvalOcrSariVatDetectResp, error) {
	response, err := ends.EvalOcrSariVatDetectEP(ctx, img)
	if err != nil {
		return EvalOcrSariVatDetectResp{}, err
	}
	resp := response.(EvalOcrSariVatDetectResp)
	return resp, err
}

//-------------------------------------------------------------------------------
//文本识别

type EvalOcrSariVatRecogReq struct {
	Data struct {
		IMG pimage.Image
	} `json:"data"`
	Params struct {
		Bboxes [][4][2]float32 `json:"bboxes"`
	} `json:"params"`
}

type EvalOcrSariVatRecogResp struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Result  struct {
		Texts []string `json:"text"`
	} `json:"result"`
}

type EvalOcrSariVatRecogService interface {
	EvalOcrSariVatRecog(ctx context.Context, img EvalOcrSariVatRecogReq) (EvalOcrSariVatRecogResp, error)
}

var _ EvalOcrSariVatRecogService = EvalOcrSariVatRecogEndpoints{}

type EvalOcrSariVatRecogEndpoints struct {
	EvalOcrSariVatRecogEP endpoint.Endpoint
}

func (ends EvalOcrSariVatRecogEndpoints) EvalOcrSariVatRecog(ctx context.Context, img EvalOcrSariVatRecogReq) (
	EvalOcrSariVatRecogResp, error) {
	response, err := ends.EvalOcrSariVatRecogEP(ctx, img)
	if err != nil {
		return EvalOcrSariVatRecogResp{}, err
	}
	resp := response.(EvalOcrSariVatRecogResp)
	return resp, err
}

//-------------------------------------------------------------------------------
//文字结构化

type EvalOcrSariVatPostProcessReq struct {
	Data struct {
		IMG pimage.Image
	} `json:"data"`
	Params struct {
		Type  string   `json:"type"`
		Texts []string `json:"texts"`
	} `json:"params"`
}

type EvalOcrSariVatPostProcessResp struct {
	Code    int                    `json:"code"`
	Message string                 `json:"message"`
	Result  map[string]interface{} `json:"result"`
}

type EvalOcrSariVatPostProcessService interface {
	EvalOcrSariVatPostProcess(ctx context.Context, img EvalOcrSariVatPostProcessReq) (EvalOcrSariVatPostProcessResp, error)
}

var _ EvalOcrSariVatPostProcessService = EvalOcrSariVatPostProcessEndpoints{}

type EvalOcrSariVatPostProcessEndpoints struct {
	EvalOcrSariVatPostProcessEP endpoint.Endpoint
}

func (ends EvalOcrSariVatPostProcessEndpoints) EvalOcrSariVatPostProcess(ctx context.Context, img EvalOcrSariVatPostProcessReq) (
	EvalOcrSariVatPostProcessResp, error) {
	response, err := ends.EvalOcrSariVatPostProcessEP(ctx, img)
	if err != nil {
		return EvalOcrSariVatPostProcessResp{}, err
	}
	resp := response.(EvalOcrSariVatPostProcessResp)
	return resp, err
}
