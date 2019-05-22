package foo

import (
	"context"

	xlog "github.com/qiniu/xlog.v1"
	"qiniu.com/argus/sdk/video"
	svideo "qiniu.com/argus/service/service/video"
	"qiniu.com/argus/service/service/video/live"
	video0 "qiniu.com/argus/video"
)

func NewOP() svideo.OPFactory {
	return live.SimpleCutOPFactory{
		NewCutsFunc: func(
			_ context.Context,
			opParams video0.OPParams,
			options ...video.CutOpOption,
		) (video.CutsPipe, error) {
			return video.CreateCutOP(
				func(ctx context.Context, cut *video.Cut) (interface{}, error) {
					xlog.FromContextSafe(ctx).Debug("foo")
					return video0.CutResultWithLabels{
						Labels: []video0.ResultLabel{{Name: "foo", Score: 1.0}},
					}, nil
				},
				options...,
			)
		},
	}
}
