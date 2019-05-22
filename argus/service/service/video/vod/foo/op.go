package foo

import (
	"context"

	"qiniu.com/argus/video/vframe"

	"qiniu.com/argus/service/service/video/vod"

	"qiniu.com/argus/sdk/video"
	svideo "qiniu.com/argus/service/service/video"
	video0 "qiniu.com/argus/video"
)

func NewOP() svideo.OPFactory {
	return vod.SimpleCutOPFactory{
		NewCutsFunc: func(
			_ context.Context,
			opParams video0.OPParams,
			_ vframe.VframeParams,
			options ...video.CutOpOption,
		) (video.CutsPipe, error) {
			return video.CreateCutOP(
				func(ctx context.Context, cut *video.Cut) (interface{}, error) {
					return video0.CutResultWithLabels{
						Labels: []video0.ResultLabel{{Name: "foo", Score: 1.0}},
					}, nil
				},
				options...,
			)
		},
	}
}
