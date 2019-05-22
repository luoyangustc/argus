package scene

import (
	"context"

	"github.com/go-kit/kit/endpoint"

	pimage "qiniu.com/argus/service/service/image"
)

type EvalSceneReq struct {
	Data struct {
		IMG pimage.Image
	} `json:"data"`
	Params struct {
		Threshold float32 `json:"threshold,omitempty"`
		Limit     int     `json:"limit,omitempty"`
	} `json:"params,omitempty"`
}

type EvalSceneResp struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Result  struct {
		Confidences []struct {
			Class   string   `json:"class"`
			Index   int      `json:"index"`
			Label   []string `json:"label"`
			Score   float32  `json:"score"`
			LabelCN string   `json:"label_cn,omitempty"`
		} `json:"confidences"`
	} `json:"result"`
}

type EvalSceneService interface {
	EvalScene(context.Context, EvalSceneReq) (EvalSceneResp, error)
}

var _ EvalSceneService = EvalSceneEndpoints{}

type EvalSceneEndpoints struct {
	EvalSceneEP endpoint.Endpoint
}

func (ends EvalSceneEndpoints) EvalScene(ctx context.Context, req EvalSceneReq) (EvalSceneResp, error) {
	response, err := ends.EvalSceneEP(ctx, req)
	if err != nil {
		return EvalSceneResp{}, err
	}
	resp := response.(EvalSceneResp)
	return resp, nil
}
