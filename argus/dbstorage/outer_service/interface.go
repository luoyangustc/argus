package outer_service

import (
	"context"

	"qiniu.com/argus/dbstorage/proto"
)

type IFaceGroup interface {
	Add(context.Context, proto.TaskConfig, proto.GroupName, proto.ImageId, proto.ImageURI, proto.ImageTag, proto.ImageDesc) (string, error)
	CreateGroup(context.Context, string) error
}
