package pulp

import (
	"context"
	"strconv"

	"github.com/go-kit/kit/endpoint"

	"qiniu.com/argus/sdk/video"
	"qiniu.com/argus/service/service/image"
	imagePulp "qiniu.com/argus/service/service/image/pulp"
	svideo "qiniu.com/argus/service/service/video"
	"qiniu.com/argus/service/service/video/ops/pulp"
	"qiniu.com/argus/service/service/video/vod"
	video0 "qiniu.com/argus/video"
	"qiniu.com/argus/video/vframe"
)

func NewOP(gen func() endpoint.Endpoint) svideo.OPFactory {
	return vod.SpecialCutOPFactory{
		VframeParamsFunc: func(_ context.Context, origin vframe.VframeParams) *vframe.VframeParams {
			if origin.GetMode() != vframe.MODE_INTERVAL {
				return nil
			}
			var interval = video0.GCD(int64(1000*origin.Interval), 500)
			return &vframe.VframeParams{
				Mode:     origin.Mode,
				Interval: float64(interval) / 1000,
			}
		},
		NewCutsFunc: func(
			_ context.Context,
			opParams video0.OPParams,
			vframeParams vframe.VframeParams,
			options ...video.CutOpOption,
		) (video.CutsPipe, error) {

			eval_ := gen()
			eval := func(ctx context.Context, body []byte) (imagePulp.PulpResult, error) {
				req := imagePulp.PulpReq{}
				req.Data.IMG.URI = image.DataURI(body)
				return PulpOP{}.DoEval(ctx, req, eval_)
			}
			var func_ func(context.Context, *video.Cut) (interface{}, error)
			var options1 []video.CutOpOption
			if vframeParams.GetMode() == vframe.MODE_KEY {
				func_, options1 = pulp.NewOPModeKey(eval)
			} else {
				func_, options1 = pulp.NewOP(eval)
			}
			options = append(options, options1...)

			func1 := func(ctx context.Context, cut *video.Cut) (interface{}, error) {
				resp, err := func_(ctx, cut)
				if err != nil {
					return resp, err
				}

				ret := video0.CutResultWithLabels{
					CutResult: video0.CutResult{
						Result: resp,
					},
				}
				ret.Labels, _ = Parse(ctx, resp.(imagePulp.PulpResult)) // TODO check
				return ret, nil
			}

			return video.CreateCutOP(func1, options...)
		},
	}
}

func Parse(ctx context.Context, r imagePulp.PulpResult) ([]video0.ResultLabel, error) {
	return []video0.ResultLabel{{Name: strconv.Itoa(r.Label), Score: r.Score}}, nil
}

type PulpOP struct {
	imagePulp.PulpService
}

func NewPulpOP(s imagePulp.PulpService) PulpOP {
	return PulpOP{PulpService: s}
}

func (op PulpOP) NewEval() endpoint.Endpoint {
	return func(ctx context.Context, req_ interface{}) (interface{}, error) {
		req := req_.(imagePulp.PulpReq)
		return op.eval(ctx, req)
	}
}
func (op PulpOP) DoEval(
	ctx context.Context, req imagePulp.PulpReq, ep endpoint.Endpoint,
) (imagePulp.PulpResult, error) {
	req.Params.Limit = 3
	resp, err := ep(ctx, req)
	if resp != nil {
		return resp.(imagePulp.PulpResult), err
	}
	return imagePulp.PulpResult{}, err
}
func (op PulpOP) eval(ctx context.Context, req imagePulp.PulpReq) (imagePulp.PulpResult, error) {
	resp, err := op.PulpService.Pulp(ctx, req)
	if err != nil {
		return imagePulp.PulpResult{}, err
	}
	return resp.Result, nil
}
