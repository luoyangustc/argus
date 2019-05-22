// +build !cublas

package gpu

import (
	"context"

	"qiniu.com/argus/feature_group_private"
	"qiniu.com/argus/feature_group_private/proto"
	"qiniu.com/argus/feature_group_private/search"
)

var _ search.Sets = new(Sets)

type Sets struct {
}

func NewSets(config search.Config) (s search.Sets, err error) {
	err = search.ErrUseGPUMode
	return
}

func (s *Sets) New(ctx context.Context, name search.SetName, config search.Config, state search.SetState) (err error) {
	err = search.ErrUseGPUMode
	return
}

func (s *Sets) Get(context.Context, search.SetName) (set search.Set, err error) {
	err = search.ErrUseGPUMode
	return
}

func (s *Sets) Delete(context.Context, search.SetName) {
}

//-------------------------- Set --------------------------//
type Set struct {
}

func (s *Set) Config(ctx context.Context) (c search.Config) {
	return
}

func (s *Set) SetState(ctx context.Context, state search.SetState) (err error) {
	err = search.ErrUseGPUMode
	return
}

func (s *Set) Destroy(ctx context.Context) (err error) {
	err = search.ErrUseGPUMode
	return
}

func (s *Set) Add(ctx context.Context, features ...proto.Feature) (err error) {
	err = search.ErrUseGPUMode
	return
}

func (s *Set) Delete(ctx context.Context, ids ...proto.FeatureID) (err error) {
	err = search.ErrUseGPUMode
	return
}

func (s *Set) Update(ctx context.Context, features ...proto.Feature) (err error) {
	err = search.ErrUseGPUMode
	return
}

func (s *Set) Search(ctx context.Context, threshold float32, limit int, features ...proto.FeatureValue) (ret [][]feature_group.FeatureSearchItem, err error) {
	err = search.ErrUseGPUMode
	return
}

func (s *Set) SpaceAvailable(ctx context.Context, size int) bool {
	return false
}

func (s *Set) Compare(ctx context.Context, threshold float32, limit int, target Set) ([]feature_group.FeatureCompareItem, error) {
	return nil, search.ErrUseGPUMode
}
