package ocr

import (
	"context"

	"github.com/go-kit/kit/endpoint"
	pimage "qiniu.com/argus/service/service/image"
)

type EvalOcrRefinedetReq struct {
	Data struct {
		IMG pimage.Image
	} `json:"data"`
}

type EvalOcrRefinedetResp struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Result  struct {
		Items []OcrClassifyResult `json:"items"`
	} `json:"result"`
}

type EvalOcrClassifyService interface {
	EvalOcrClassify(ctx context.Context, img EvalOcrRefinedetReq) (eocresp EvalOcrRefinedetResp, err error)
}

var _ EvalOcrClassifyService = EvalOcrClassifyEndpoints{}

type EvalOcrClassifyEndpoints struct {
	EvalOcrClassifyEP endpoint.Endpoint
}

func (ends EvalOcrClassifyEndpoints) EvalOcrClassify(ctx context.Context, img EvalOcrRefinedetReq) (
	EvalOcrRefinedetResp, error) {
	response, err := ends.EvalOcrClassifyEP(ctx, img)
	if err != nil {
		return EvalOcrRefinedetResp{}, err
	}
	resp := response.(EvalOcrRefinedetResp)
	return resp, nil
}

//-----------------------------------------------------------------------------------------------------------------
type EvalOcrTerrorReq struct {
	Data struct {
		IMG pimage.Image
	} `json:"data"`
}

type EvalOcrTerrorResp struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Result  struct {
		Confidences []struct {
			Index int     `json:"index"`
			Class string  `json:"class"`
			Score float32 `json:"score"`
		} `json:"confidences"`
	} `json:"result"`
}

type EvalOcrTerrorService interface {
	EvalOcrTerror(ctx context.Context, img EvalOcrTerrorReq) (eotr EvalOcrTerrorResp, err error)
}

var _ EvalOcrTerrorService = EvalOcrTerrorEndpoints{}

type EvalOcrTerrorEndpoints struct {
	EvalOcrTerrorEP endpoint.Endpoint
}

func (ends EvalOcrTerrorEndpoints) EvalOcrTerror(ctx context.Context, img EvalOcrTerrorReq) (
	EvalOcrTerrorResp, error) {
	response, err := ends.EvalOcrTerrorEP(ctx, img)
	if err != nil {
		return EvalOcrTerrorResp{}, err
	}
	resp := response.(EvalOcrTerrorResp)
	return resp, nil
}
