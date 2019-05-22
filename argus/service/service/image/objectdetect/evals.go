package objectdetect

import (
	"context"

	"github.com/go-kit/kit/endpoint"
	pimage "qiniu.com/argus/service/service/image"
)

type EvalObjectDetectReq struct {
	Data struct {
		IMG pimage.Image
	} `json:"data"`
	Params struct {
		Threshold float32 `json:"threshold,omitempty"`
	} `json:"params,omitempty"`
}

type EvalObjectDetect struct {
	Index   int      `json:"index"`
	Class   string   `json:"class"`
	Score   float32  `json:"score"`
	Pts     [][2]int `json:"pts"`
	LabelCN string   `json:"label_cn,omitempty"`
}

type EvalObjectDetectResp struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Result  struct {
		Detections []EvalObjectDetect `json:"detections"`
	} `json:"result"`
}

type EvalObjectDetectService interface {
	EvalObjectDetect(ctx context.Context, img EvalObjectDetectReq) (etcresp EvalObjectDetectResp, err error)
}

var _ EvalObjectDetectService = EvalObjectDetectEndpoints{}

type EvalObjectDetectEndpoints struct {
	EvalObjectDetectEP endpoint.Endpoint
}

func (ends EvalObjectDetectEndpoints) EvalObjectDetect(
	ctx context.Context, img EvalObjectDetectReq) (EvalObjectDetectResp, error) {
	response, err := ends.EvalObjectDetectEP(ctx, img)
	if err != nil {
		return EvalObjectDetectResp{}, err
	}
	resp := response.(EvalObjectDetectResp)
	return resp, nil
}
