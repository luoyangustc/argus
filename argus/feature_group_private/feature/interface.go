/*
Package image_feature 提供图片特征、人脸特征服务的client
*/
package feature

import (
	"context"

	feature_group "qiniu.com/argus/feature_group_private"
	"qiniu.com/argus/feature_group_private/proto"
)

type BaseFeature interface {
	CreateGroup(context.Context, proto.NodeAddress, proto.GroupName, proto.GroupConfig) error
	RemoveGroup(context.Context, proto.NodeAddress, proto.GroupName) error
	AddFeature(context.Context, proto.NodeAddress, proto.GroupName, ...proto.Feature) error
	DeleteFeature(context.Context, proto.NodeAddress, proto.GroupName, ...proto.FeatureID) error
	UpdateFeature(context.Context, proto.NodeAddress, proto.GroupName, ...proto.Feature) error
	SearchFeature(context.Context, proto.NodeAddress, proto.GroupName, float32, int, ...proto.FeatureValue) ([][]feature_group.FeatureSearchItem, error)
}

type ImageFeature interface {
	Image(context.Context, proto.ImageURI) (proto.FeatureValue, error)
}

type FaceFeature interface {
	Face(context.Context, proto.ImageURI, [][2]int) (proto.FeatureValue, error)
	FaceBoxesQuality(context.Context, proto.ImageURI) (
		[]proto.FaceDetectBox,
		error,
	)
	FaceBoxes(context.Context, proto.ImageURI) ([]proto.BoundingBox, error)
}
