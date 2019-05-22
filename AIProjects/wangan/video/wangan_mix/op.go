package wangan_mix

import (
	"context"
	"encoding/json"

	"github.com/go-kit/kit/endpoint"

	simage "qiniu.com/argus/AIProjects/wangan/image/wangan_mix"
	"qiniu.com/argus/sdk/video"
	. "qiniu.com/argus/service/service"
	"qiniu.com/argus/service/service/image"
	svideo "qiniu.com/argus/service/service/video"
	"qiniu.com/argus/service/service/video/vod"
	video0 "qiniu.com/argus/video"
	"qiniu.com/argus/video/vframe"
)

type WanganMixResult struct {
	Label       int                `json:"label"`
	Score       float32            `json:"score"`
	Classes     []string           `json:"classes,omitempty"`
	ClassScores map[string]float32 `json:"-"`
}

func NewOP(gen func() endpoint.Endpoint) svideo.OPFactory {
	return vod.SimpleCutOPFactory{
		NewCutsFunc: func(
			ctx context.Context,
			opParams video0.OPParams,
			_ vframe.VframeParams,
			options ...video.CutOpOption) (video.CutsPipe, error) {
			eval := gen()
			params_, err := ParseParams(opParams)
			if err != nil {
				return nil, ErrArgs(err.Error())
			}
			params := params_.(struct {
				Detail bool   `json:"detail"`
				Type   string `json:"type"`
			})
			func_ := func(ctx context.Context, cut *video.Cut) (interface{}, error) {
				req := simage.WanganMixReq{}
				body, _ := cut.Body()
				req.Data.IMG.URI = image.DataURI(body)
				req.Params = params
				resp, err := WanganMixOP{}.DoEval(ctx, req, eval)
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

func Parse(ctx context.Context, r WanganMixResult) ([]video0.ResultLabel, error) {
	ret := make([]video0.ResultLabel, 0)
	for class, score := range r.ClassScores {
		ret = append(ret, video0.ResultLabel{
			Name:  class,
			Score: score,
		})
	}
	return ret, nil
}

func ParseParams(params video0.OPParams) (interface{}, error) {
	var req0 simage.WanganMixReq
	{
		bs, _ := json.Marshal(params.Other)
		err := json.Unmarshal(bs, &req0.Params)
		if err != nil {
			return nil, err
		}
	}
	return req0.Params, nil
}

type WanganMixService interface {
	WanganMix(context.Context, simage.WanganMixReq) (WanganMixResult, error)
}

type WanganMixOP struct {
	WanganMixService
}

func NewWanganMixOP(s WanganMixService) WanganMixOP {
	return WanganMixOP{WanganMixService: s}
}

func (op WanganMixOP) DoEval(
	ctx context.Context, req simage.WanganMixReq, ep endpoint.Endpoint,
) (WanganMixResult, error) {
	resp, err := ep(ctx, req)
	if resp != nil {
		return resp.(WanganMixResult), err
	}
	return WanganMixResult{}, err
}

func (op WanganMixOP) NewEval() endpoint.Endpoint {
	return func(ctx context.Context, req_ interface{}) (interface{}, error) {
		req := req_.(simage.WanganMixReq)
		return op.WanganMixService.WanganMix(ctx, req)
	}
}
