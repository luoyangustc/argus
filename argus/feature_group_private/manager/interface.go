package manager

import (
	"context"
	"errors"

	"qiniu.com/argus/feature_group_private/proto"
)

var (
	ErrInvalidGroupParams = errors.New("invalid group params")
	ErrGroupExist         = errors.New("group is already exist")
	ErrGroupNotExist      = errors.New("group is not exist")
	ErrFeatureExist       = errors.New("feature is already exist")
	ErrFeatureNotExist    = errors.New("feature is not exist")
)

type Groups interface {
	New(context.Context, proto.GroupName, proto.GroupConfig) error
	Get(context.Context, proto.GroupName) (Group, error)
	All(context.Context) ([]proto.GroupName, error)

	AllNodes(context.Context) ([]proto.Node, error)
	UpsertNode(context.Context, proto.Node) error
}

type Group interface {
	Destroy(context.Context) error
	Count(context.Context, proto.HashKeyRange) (int, error)
	CountTags(context.Context) (int, error)
	CountWithoutHashKey(context.Context) (int, error)
	Config(context.Context) proto.GroupConfig

	Get(context.Context, proto.FeatureID) (proto.Feature, error)
	Exist(context.Context, ...proto.FeatureID) ([]proto.FeatureID, error)
	Add(context.Context, ...proto.Feature) error
	Delete(context.Context, ...proto.FeatureID) error
	Update(context.Context, ...proto.Feature) error
	Tags(context.Context, string, int) ([]proto.GroupTagInfo, string, error)
	FilterByTag(ctx context.Context, tag proto.FeatureTag, marker string, limit int) ([]proto.Feature, string, error)

	Iter(context.Context, proto.HashKeyRange, func(context.Context, ...proto.Feature) error) error
	EnsureHashKey(context.Context, func(proto.FeatureID) proto.FeatureHashKey) error
}
