package vod

import (
	"context"

	"qiniu.com/argus/sdk/video"
	svideo "qiniu.com/argus/service/service/video"
	video0 "qiniu.com/argus/video"
	"qiniu.com/argus/video/vframe"
)

type OPs map[string]svideo.OPFactory
type CutOP interface {
	svideo.OP
	NewCuts(context.Context, vframe.VframeParams, ...video.CutOpOption) (video.CutsPipe, error)
}
type SpecialCutOP interface {
	CutOP
	VframeParams(context.Context, vframe.VframeParams) *vframe.VframeParams
}

////////////////////////////////////////////////////////////////////////////////

var _ svideo.OPFactory = SimpleCutOPFactory{}

type SimpleCutOPFactory struct {
	NewCutsFunc func(
		context.Context,
		video0.OPParams,
		vframe.VframeParams,
		...video.CutOpOption) (video.CutsPipe, error)
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
	ctx context.Context,
	vframeParams vframe.VframeParams,
	options ...video.CutOpOption) (video.CutsPipe, error) {
	return op.NewCutsFunc(ctx, op.OPParams, vframeParams, options...)
}

////////////////////////////////////////////////////////////////////////////////

var _ svideo.OPFactory = SimpleCutOPFactory{}

type SpecialCutOPFactory struct {
	VframeParamsFunc func(
		context.Context, vframe.VframeParams,
	) *vframe.VframeParams
	NewCutsFunc func(
		context.Context, video0.OPParams, vframe.VframeParams, ...video.CutOpOption,
	) (video.CutsPipe, error)
}

func (of SpecialCutOPFactory) Count() int32                       { return video0.MAX_OP_COUNT }
func (of SpecialCutOPFactory) Release(context.Context, svideo.OP) {}
func (of SpecialCutOPFactory) Create(_ context.Context, params video0.OPParams) (svideo.OP, error) {
	return specialCutOP{OPParams: params, SpecialCutOPFactory: of}, nil
}

var _ SpecialCutOP = specialCutOP{}

type specialCutOP struct {
	video0.OPParams
	SpecialCutOPFactory
}

func (op specialCutOP) Reset(context.Context) error { return nil }
func (op specialCutOP) Params() video0.OPParams     { return op.OPParams }
func (op specialCutOP) NewCuts(
	ctx context.Context,
	vframeParams vframe.VframeParams,
	options ...video.CutOpOption) (video.CutsPipe, error) {
	return op.NewCutsFunc(ctx, op.OPParams, vframeParams, options...)
}
func (op specialCutOP) VframeParams(
	ctx context.Context, params vframe.VframeParams,
) *vframe.VframeParams {
	return op.VframeParamsFunc(ctx, params)
}
