package ocrbankcard

import (
	"context"

	"github.com/go-kit/kit/endpoint"
	pimage "qiniu.com/argus/service/service/image"
)

type EvalOcrSariBankcardDetectReq struct {
	Data struct {
		IMG pimage.Image
	} `json:"data"`
}

type EvalOcrSariBankcardDetectResp struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Result  struct {
		Bboxes [][4][2]int `json:"bboxes"`
	} `json:"result"`
}

type EvalOcrSariBankcardDetectService interface {
	EvalOcrSariBankcardDetect(ctx context.Context, img EvalOcrSariBankcardDetectReq) (edresp EvalOcrSariBankcardDetectResp, err error)
}

var _ EvalOcrSariBankcardDetectService = EvalOcrSariBankcardDetectEndpoints{}

type EvalOcrSariBankcardDetectEndpoints struct {
	EvalOcrSariBankcardDetectEP endpoint.Endpoint
}

func (ends EvalOcrSariBankcardDetectEndpoints) EvalOcrSariBankcardDetect(ctx context.Context, img EvalOcrSariBankcardDetectReq) (
	EvalOcrSariBankcardDetectResp, error) {
	response, err := ends.EvalOcrSariBankcardDetectEP(ctx, img)
	if err != nil {
		return EvalOcrSariBankcardDetectResp{}, err
	}
	resp := response.(EvalOcrSariBankcardDetectResp)
	return resp, err
}

//-------------------------------------------------------------------------------------

type EvalOcrSariBankcardRecogReq struct {
	Data struct {
		IMG pimage.Image
	} `json:"data"`
	Params struct {
		Bboxes [][4][2]int `json:"bboxes"`
	} `json:"params"`
}

type EvalOcrSariBankcardRecogResp struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Result  struct {
		Texts []string `json:"text"`
	} `json:"result"`
}

type EvalOcrSariBankcardRecogService interface {
	EvalOcrSariBankcardRecog(ctx context.Context, img EvalOcrSariBankcardRecogReq) (EvalOcrSariBankcardRecogResp, error)
}

var _ EvalOcrSariBankcardRecogService = EvalOcrSariBankcardRecogEndpoints{}

type EvalOcrSariBankcardRecogEndpoints struct {
	EvalOcrSariBankcardRecogEP endpoint.Endpoint
}

func (ends EvalOcrSariBankcardRecogEndpoints) EvalOcrSariBankcardRecog(ctx context.Context, img EvalOcrSariBankcardRecogReq) (
	EvalOcrSariBankcardRecogResp, error) {
	response, err := ends.EvalOcrSariBankcardRecogEP(ctx, img)
	if err != nil {
		return EvalOcrSariBankcardRecogResp{}, err
	}

	resp := response.(EvalOcrSariBankcardRecogResp)
	return resp, err
}
