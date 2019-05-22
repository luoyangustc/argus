package video

import (
	"context"

	"qiniu.com/argus/video"
)

type OPs map[string]interface{}
type OPFactory interface {
	Count() int32
	Create(context.Context, video.OPParams) (OP, error)
	Release(context.Context, OP)
}

type OP interface {
	Reset(context.Context) error
	Params() video.OPParams
}
