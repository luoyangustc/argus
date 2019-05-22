package politician

import (
	"context"

	"qiniu.com/argus/video/vframe"

	"github.com/go-kit/kit/endpoint"

	"qiniu.com/argus/sdk/video"
	"qiniu.com/argus/service/service/image"
	"qiniu.com/argus/service/service/image/politician"
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
			func_ := func(ctx context.Context, cut *video.Cut) (interface{}, error) {
				req := politician.Req{}
				body, _ := cut.Body()
				req.Data.IMG.URI = image.DataURI(body)
				resp, err := PoliticianOP{}.DoEval(ctx, req, eval)
				if err != nil {
					return resp, err
				}

				ret := video0.CutResultWithLabels{}
				ret.Result = resp
				ret.Labels, _ = Parse(ctx, resp)
				return ret, nil
			}
			return video.CreateCutOP(func_, options...)
		},
	}
}

func Parse(ctx context.Context, r FaceSearchResult) ([]video0.ResultLabel, error) {
	ret := make([]video0.ResultLabel, 0)
	for _, v := range r.Detections {
		if len(v.Value.Name) > 0 {
			ret = append(ret, video0.ResultLabel{
				Name:  v.Value.Name,
				Score: v.Value.Score,
			})
		}
	}
	return ret, nil
}

type PoliticianOP struct {
	politician.FaceSearchService
}

type FaceSearchResult struct {
	Detections []politician.FaceSearchDetail `json:"detections"`
}

func NewPoliticianOP(s politician.FaceSearchService) PoliticianOP {
	return PoliticianOP{FaceSearchService: s}
}
func (op PoliticianOP) NewEval() endpoint.Endpoint {
	return func(ctx context.Context, req_ interface{}) (interface{}, error) {
		req := req_.(politician.Req)
		return op.eval(ctx, req)
	}
}
func (op PoliticianOP) DoEval(
	ctx context.Context, req politician.Req, ep endpoint.Endpoint,
) (FaceSearchResult, error) {
	resp, err := ep(ctx, req)
	if resp != nil {
		return resp.(FaceSearchResult), err
	}
	return FaceSearchResult{}, err
}
func (op PoliticianOP) eval(ctx context.Context, req politician.Req) (FaceSearchResult, error) {
	resp, err := op.FaceSearchService.FaceSearch(ctx, req)
	if err != nil {
		return FaceSearchResult{}, err
	}
	return FaceSearchResult{Detections: resp.Result.Detections}, nil
}
