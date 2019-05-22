package censor

import (
	"context"

	"qiniu.com/argus/sdk/video"
)

type OPs map[string]OPFactory

type OPFactory interface {
	Create(context.Context) (OP, error)
}

type OP interface {
	NewCuts(context.Context, CutParam, ...video.CutOpOption) (video.CutsPipe, error)
}

type SpecialOP interface {
	OP
	CutParam(context.Context, CutParam) *CutParam
}

////////////////////////////////////////////////////////////////////////////////

var _ OPFactory = SimpleCutOPFactory{}

type SimpleCutOPFactory struct {
	NewCutsFunc func(context.Context, CutParam, ...video.CutOpOption) (video.CutsPipe, error)
}

func (of SimpleCutOPFactory) Create(_ context.Context) (OP, error) {
	return simpleCutOP{SimpleCutOPFactory: of}, nil
}

var _ OP = simpleCutOP{}

type simpleCutOP struct {
	SimpleCutOPFactory
}

func (op simpleCutOP) NewCuts(
	ctx context.Context,
	cutParma CutParam,
	options ...video.CutOpOption) (video.CutsPipe, error) {
	return op.NewCutsFunc(ctx, cutParma, options...)
}

////////////////////////////////////////////////////////////////////////////////

var _ OPFactory = SimpleCutOPFactory{}

type SpecialCutOPFactory struct {
	CutParamFunc func(
		context.Context, CutParam,
	) *CutParam
	NewCutsFunc func(
		context.Context, CutParam, ...video.CutOpOption,
	) (video.CutsPipe, error)
}

func (of SpecialCutOPFactory) Create(_ context.Context) (OP, error) {
	return specialCutOP{SpecialCutOPFactory: of}, nil
}

var _ SpecialOP = specialCutOP{}

type specialCutOP struct {
	SpecialCutOPFactory
}

func (op specialCutOP) NewCuts(
	ctx context.Context,
	cutParma CutParam,
	options ...video.CutOpOption) (video.CutsPipe, error) {
	return op.NewCutsFunc(ctx, cutParma, options...)
}
func (op specialCutOP) CutParam(
	ctx context.Context, cutParam CutParam,
) *CutParam {
	return op.CutParamFunc(ctx, cutParam)
}
