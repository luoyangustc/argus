package ads

import (
	"context"

	"github.com/go-kit/kit/endpoint"

	"qiniu.com/argus/sdk/video"
	"qiniu.com/argus/service/service/image"
	pimage "qiniu.com/argus/service/service/image"
	"qiniu.com/argus/service/service/image/ads"
	"qiniu.com/argus/service/service/video/censor"
)

func NewOP(gen func() endpoint.Endpoint) censor.OPFactory {
	return censor.SimpleCutOPFactory{
		NewCutsFunc: func(
			ctx context.Context,
			_ censor.CutParam,
			options ...video.CutOpOption,
		) (video.CutsPipe, error) {

			eval := gen()
			func_ := func(ctx context.Context, cut *video.Cut) (interface{}, error) {
				req := pimage.ImageCensorReq{}
				body, _ := cut.Body()
				req.Data.IMG.URI = image.DataURI(body)

				resp, err := eval(ctx, req)
				if err != nil {
					return nil, err
				}

				return resp, nil
			}
			return video.CreateCutOP(func_, options...)
		},
	}
}

type AdsOP struct {
	ads.AdsService
}

func NewAdsOP(s ads.AdsService) AdsOP {
	return AdsOP{AdsService: s}
}

func (op AdsOP) NewEval() endpoint.Endpoint {
	return func(ctx context.Context, req_ interface{}) (interface{}, error) {
		req := req_.(pimage.ImageCensorReq)
		return op.eval(ctx, req)
	}
}

func (op AdsOP) eval(ctx context.Context, req pimage.ImageCensorReq) (pimage.SceneResult, error) {
	resp, err := op.AdsService.AdsCensor(ctx, req)
	if err != nil {
		return pimage.SceneResult{}, err
	}
	return resp, nil
}
