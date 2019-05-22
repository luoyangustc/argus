package search

import (
	"context"

	"qiniu.com/argus/feature_group_private"
	"qiniu.com/argus/feature_group_private/proto"
)

type SetName string

type SetState uint32

type Config struct {
	Dimension int    `json:"dimension"`
	Precision int    `json:"precision"`
	DeviceID  int    `json:"device_id"`
	Capacity  int    `json:"capacity"`
	BlockSize int    `json:"block_size"`
	BlockNum  int    `json:"block_num"`
	BatchSize int    `json:"batch_size"`
	Version   uint64 `json:"version"`
}

// Sets ...
type Sets interface {
	New(context.Context, SetName, Config, SetState) error
	Get(context.Context, SetName) (Set, error)
	Delete(context.Context, SetName)
}

// Set ...
type Set interface {
	SetState(context.Context, SetState) error
	Destroy(context.Context) error
	Config(context.Context) Config

	Add(context.Context, ...proto.Feature) error
	Delete(context.Context, ...proto.FeatureID) ([]proto.FeatureID, error)
	Update(context.Context, ...proto.Feature) error
	SpaceAvailable(context.Context, int) bool
	Get(context.Context, proto.FeatureID) (proto.FeatureValue, error)

	Search(ctx context.Context,
		threshold float32, limit int,
		features ...proto.FeatureValue,
	) (
		[][]feature_group.FeatureSearchItem,
		error,
	)
	Compare(ctx context.Context,
		threshold float32, limit int,
		target Set,
	) (
		[]feature_group.FeatureCompareItem,
		error,
	)
}
