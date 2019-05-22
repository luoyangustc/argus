package terror

import (
	"context"

	"github.com/go-kit/kit/endpoint"

	"qiniu.com/argus/service/service/biz"
	pimage "qiniu.com/argus/service/service/image"
	"qiniu.com/argus/utility/evals"
)

//----------------------------------------------------------------------------//

type SimpleReq struct {
	Data struct {
		// URI string `json:"uri"`
		IMG pimage.Image
	} `json:"data"`
}

type TerrorMixupResp struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Result  struct {
		Checkpoint  string `json:"checkpoint"`
		Confidences []struct {
			Index int     `json:"index"`
			Class string  `json:"class"`
			Score float32 `json:"score"`
		} `json:"confidences"`
	} `json:"result"`
}

//----------------------------------------------------------------------------//

type EvalTerrorDetectService interface {
	EvalTerrorDetect(ctx context.Context, img SimpleReq) (evals.TerrorDetectResp, error)
}

var _ EvalTerrorDetectService = EvalTerrorDetectEndpoints{}

type EvalTerrorDetectEndpoints struct {
	EvalTerrorDetectEP endpoint.Endpoint
}

func (ends EvalTerrorDetectEndpoints) EvalTerrorDetect(
	ctx context.Context, img SimpleReq,
) (evals.TerrorDetectResp, error) {
	response, err := ends.EvalTerrorDetectEP(ctx, img)
	if err != nil {
		return evals.TerrorDetectResp{}, err
	}
	resp := response.(evals.TerrorDetectResp)
	return resp, nil
}

var EVAL_TERROR_DET_CONFIG = biz.EvalModelConfig{
	Image: "reg.qiniu.com/avaprd/aisdk-terror_detect:20190305-v230-CENSORv3.5.1",
	Type:  biz.EvalRunTypeSDK,
}

//----------------------------------------------------------------------------//
type EvalTerrorMixupService interface {
	EvalTerrorMixup(ctx context.Context, img SimpleReq) (TerrorMixupResp, error)
}

var _ EvalTerrorMixupService = EvalTerrorMixupEndpoints{}

type EvalTerrorMixupEndpoints struct {
	EvalTerrorMixupEP endpoint.Endpoint
}

func (ends EvalTerrorMixupEndpoints) EvalTerrorMixup(
	ctx context.Context, img SimpleReq,
) (TerrorMixupResp, error) {
	response, err := ends.EvalTerrorMixupEP(ctx, img)
	if err != nil {
		return TerrorMixupResp{}, err
	}
	resp := response.(TerrorMixupResp)
	return resp, nil
}

var EVAL_TERROR_MIXUP_CONFIG = biz.EvalModelConfig{
	Image: "reg.qiniu.com/avaprd/aisdk-terror_mixup:20190305-v230-CENSORv3.5.1",
	Type:  biz.EvalRunTypeSDK,
}
