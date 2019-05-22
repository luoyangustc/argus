package live

import (
	"context"

	"qiniu.com/argus/sdk/video"
	svideo "qiniu.com/argus/service/service/video"
	video0 "qiniu.com/argus/video"
)

type OPs map[string]svideo.OPFactory

type CutOP interface {
	svideo.OP
	NewCuts(context.Context, ...video.CutOpOption) (video.CutsPipe, error)
}

////////////////////////////////////////////////////////////////////////////////

var _ svideo.OPFactory = SimpleCutOPFactory{}

type SimpleCutOPFactory struct {
	NewCutsFunc func(
		context.Context, video0.OPParams, ...video.CutOpOption,
	) (video.CutsPipe, error)
}

func (of SimpleCutOPFactory) Count() int32                       { return video0.MAX_OP_COUNT }
func (of SimpleCutOPFactory) Release(context.Context, svideo.OP) {}
func (of SimpleCutOPFactory) Create(_ context.Context, params video0.OPParams) (svideo.OP, error) {
	return simpleCutOP{OPParams: params, SimpleCutOPFactory: of}, nil
}

var _ CutOP = simpleCutOP{}

type simpleCutOP struct {
	video0.OPParams
	SimpleCutOPFactory
}

func (op simpleCutOP) Reset(context.Context) error { return nil }
func (op simpleCutOP) Params() video0.OPParams     { return op.OPParams }
func (op simpleCutOP) NewCuts(
	ctx context.Context, options ...video.CutOpOption,
) (video.CutsPipe, error) {
	return op.NewCutsFunc(ctx, op.OPParams, options...)
}
