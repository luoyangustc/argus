package source

import (
	"context"

	"qiniu.com/argus/dbstorage/proto"
)

type ISource interface {
	Read(context.Context, func(int) proto.ImageProcess) (<-chan proto.TaskSource, error)
	GetInfo(context.Context) (int, error)
}
