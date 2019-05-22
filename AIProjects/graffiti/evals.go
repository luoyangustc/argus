package graffiti

import (
	"context"

	"github.com/go-kit/kit/endpoint"
	pimage "qiniu.com/argus/service/service/image"
)

type GraffitiDetectReq struct {
	Data struct {
		IMG pimage.Image
	} `json:"data"`
}

type GraffitiDetectResult struct {
	Index int      `json:"index"`
	Class string   `json:"class"`
	Score float32  `json:"score"`
	Pts   [][2]int `json:"pts"`
}

type GraffitiDetectResp struct {
	Code    int                    `json:"code"`
	Message string                 `json:"message"`
	Result  []GraffitiDetectResult `json:"result"`
}

type EvalGraffitiDetectService interface {
	EvalGraffitiDetect(context.Context, GraffitiDetectReq) (GraffitiDetectResp, error)
}

type EvalGraffitiDetectEndPoints struct {
	EvalGraffitiDetectEP endpoint.Endpoint
}

var _ EvalGraffitiDetectService = EvalGraffitiDetectEndPoints{}

func (ends EvalGraffitiDetectEndPoints) EvalGraffitiDetect(ctx context.Context, img GraffitiDetectReq) (GraffitiDetectResp, error) {
	resp, err := ends.EvalGraffitiDetectEP(ctx, img)
	if err != nil {
		return GraffitiDetectResp{}, err
	}
	return resp.(GraffitiDetectResp), nil
}
