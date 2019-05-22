package pulp

import (
	"context"

	"github.com/go-kit/kit/endpoint"

	"qiniu.com/argus/service/service/biz"
	"qiniu.com/argus/utility/evals"
)

//----------------------------------------------------------------------------//

type EvalPulpService interface {
	EvalPulp(ctx context.Context, req PulpReq) (evals.PulpResp, error)
}

var _ EvalPulpService = EvalPulpEndpoints{}

type EvalPulpEndpoints struct {
	EvalPulpEP endpoint.Endpoint
}

func (ends EvalPulpEndpoints) EvalPulp(ctx context.Context, req PulpReq) (evals.PulpResp, error) {
	response, err := ends.EvalPulpEP(ctx, req)
	if err != nil {
		return evals.PulpResp{}, err
	}
	resp := response.(evals.PulpResp)
	return resp, nil
}

var EVAL_PULP_CONFIG = biz.EvalModelConfig{
	Image: "reg.qiniu.com/avaprd/aisdk-pulp:20190214-v209-CENSORv3.3.3",
	Type:  biz.EvalRunTypeSDK,
}

//--------------------------------------------------------------------------------------------
type EvalPulpFilterService interface {
	EvalPulpFilter(ctx context.Context, req PulpReq) (evals.PulpResp, error)
}

var _ EvalPulpFilterService = EvalPulpFilterEndpoints{}

type EvalPulpFilterEndpoints struct {
	EvalPulpFilterEP endpoint.Endpoint
}

func (ends EvalPulpFilterEndpoints) EvalPulpFilter(ctx context.Context, req PulpReq) (evals.PulpResp, error) {
	response, err := ends.EvalPulpFilterEP(ctx, req)
	if err != nil {
		return evals.PulpResp{}, err
	}
	resp := response.(evals.PulpResp)
	return resp, nil
}

var EVAL_PULP_Filter_CONFIG = biz.EvalModelConfig{
	Image: "reg.qiniu.com/avaprd/aisdk-pulp_filter:20190214-v209-CENSORv3.3.3",
	Type:  biz.EvalRunTypeSDK,
}
