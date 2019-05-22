package police

import (
	"context"

	"github.com/go-kit/kit/endpoint"
	"qiniu.com/argus/AIProjects/wangan/image/police"
	"qiniu.com/argus/sdk/video"
	"qiniu.com/argus/service/service/image"
	svideo "qiniu.com/argus/service/service/video"
	"qiniu.com/argus/service/service/video/vod"
	video0 "qiniu.com/argus/video"
	"qiniu.com/argus/video/vframe"
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
				req := police.PoliceReq{}
				body, _ := cut.Body()
				req.Data.IMG.URI = image.DataURI(body)
				req.Params.Detail = true
				resp, err := PoliceOP{}.DoEval(ctx, req, eval)
				if err != nil {
					return resp, err
				}
				classes0 := make(map[string]bool, 0)
				for _, detect := range resp.Result.Detections {
					classes0[detect.Class] = true
				}
				var classes []string
				for class, _ := range classes0 {
					classes = append(classes, class)
				}

				ret := video0.CutResultWithLabels{
					CutResult: video0.CutResult{
						Result: struct {
							Label   int      `json:"label"`
							Score   float32  `json:"score"`
							Classes []string `json:"classes,omitempty"`
						}{
							Label:   resp.Result.Label,
							Score:   resp.Result.Score,
							Classes: classes,
						},
					},
				}
				ret.Labels, _ = Parse(ctx, resp)
				return ret, nil
			}
			return video.CreateCutOP(func_, options...)
		},
	}
}

func Parse(ctx context.Context, resp police.PoliceResp) (ret []video0.ResultLabel, err error) {
	if resp.Result.Label == 0 {
		return []video0.ResultLabel{video0.ResultLabel{Name: "normal", Score: resp.Result.Score}}, nil
	}
	for _, detect := range resp.Result.Detections {
		ret = append(ret, video0.ResultLabel{
			Name:  detect.Class,
			Score: detect.Score,
		})
	}
	return ret, nil
}

type PoliceOP struct {
	police.PoliceService
}

func NewPoliceOP(s police.PoliceService) PoliceOP { return PoliceOP{PoliceService: s} }

func (op PoliceOP) NewEval() endpoint.Endpoint {
	return func(ctx context.Context, req0 interface{}) (interface{}, error) {
		req := req0.(police.PoliceReq)
		return op.eval(ctx, req)
	}
}

func (op PoliceOP) DoEval(ctx context.Context, req police.PoliceReq, ep endpoint.Endpoint) (police.PoliceResp, error) {
	resp, err := ep(ctx, req)
	if err != nil {
		return police.PoliceResp{}, err
	}
	return resp.(police.PoliceResp), nil
}

func (op PoliceOP) eval(ctx context.Context, req police.PoliceReq) (police.PoliceResp, error) {
	return op.Police(ctx, req)
}
