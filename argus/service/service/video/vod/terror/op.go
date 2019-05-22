package terror

import (
	"context"
	"encoding/json"
	"strconv"

	"qiniu.com/argus/video/vframe"

	"github.com/go-kit/kit/endpoint"

	"qiniu.com/argus/sdk/video"
	. "qiniu.com/argus/service/service"
	"qiniu.com/argus/service/service/image"
	"qiniu.com/argus/service/service/image/terror"
	svideo "qiniu.com/argus/service/service/video"
	"qiniu.com/argus/service/service/video/vod"
	video0 "qiniu.com/argus/video"
)

func NewOP(gen func() endpoint.Endpoint) svideo.OPFactory {
	return vod.SimpleCutOPFactory{
		NewCutsFunc: func(
			ctx context.Context,
			opParams video0.OPParams,
			_ vframe.VframeParams,
			options ...video.CutOpOption,
		) (video.CutsPipe, error) {

			eval := gen()
			params_, err := ParseParams(opParams)
			if err != nil {
				return nil, ErrArgs(err.Error())
			}
			params := params_.(struct {
				Detail bool `json:"detail"`
			})

			func_ := func(ctx context.Context, cut *video.Cut) (interface{}, error) {
				req := terror.TerrorReq{}
				body, _ := cut.Body()
				req.Data.IMG.URI = image.DataURI(body)
				req.Params = params
				resp, err := TerrorOP{}.DoEval(ctx, req, eval)
				if err != nil {
					return resp, err
				}

				ret := video0.CutResultWithLabels{
					CutResult: video0.CutResult{
						Result: resp,
					},
				}
				ret.Labels, _ = Parse(ctx, resp)
				return ret, nil
			}
			return video.CreateCutOP(func_, options...)
		},
	}
}

func Parse(ctx context.Context, r terror.TerrorResult) ([]video0.ResultLabel, error) {
	return []video0.ResultLabel{{Name: strconv.Itoa(r.Label), Score: r.Score}}, nil
}

func ParseParams(params video0.OPParams) (interface{}, error) {
	var req0 terror.TerrorReq
	{
		bs, _ := json.Marshal(params.Other)
		err := json.Unmarshal(bs, &req0.Params)
		if err != nil {
			return nil, err
		}
	}
	return req0.Params, nil
}

type TerrorOP struct {
	terror.TerrorService
}

func NewTerrorOP(s terror.TerrorService) TerrorOP { return TerrorOP{TerrorService: s} }
func (op TerrorOP) NewEval() endpoint.Endpoint {
	return func(ctx context.Context, req0 interface{}) (interface{}, error) {
		req := req0.(terror.TerrorReq)
		return op.eval(ctx, req)
	}
}
func (op TerrorOP) DoEval(
	ctx context.Context, req terror.TerrorReq, ep endpoint.Endpoint,
) (terror.TerrorResult, error) {
	resp, err := ep(ctx, req)
	if resp != nil {
		return resp.(terror.TerrorResult), err
	}
	return terror.TerrorResult{}, err
}
func (op TerrorOP) eval(ctx context.Context, req terror.TerrorReq) (terror.TerrorResult, error) {
	resp, err := op.TerrorService.Terror(ctx, req)
	if err != nil {
		return terror.TerrorResult{}, err
	}
	return resp.Result, nil
}
