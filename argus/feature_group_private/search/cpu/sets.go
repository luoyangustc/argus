package cpu

import (
	"context"
	"sync"

	"qiniu.com/argus/feature_group_private/search"
)

type _CPUSets struct {
	search.Config
	*Cache
	FeatureSets map[search.SetName]search.Set
	Mutex       sync.Mutex
}

var _ search.Sets = &_CPUSets{}

func NewSets(c search.Config) (*_CPUSets, error) {
	sets := &_CPUSets{
		Config:      c,
		FeatureSets: make(map[search.SetName]search.Set, 0),
	}

	var err error
	if sets.Cache, err = NewCache(c.BlockNum, c.BlockSize); err != nil {
		return nil, err
	}

	return sets, nil
}

func (s *_CPUSets) New(ctx context.Context, name search.SetName, cfg search.Config, state search.SetState) (err error) {
	var set search.Set
	if set, err = NewSet(s.Cache, name, cfg.Dimension, cfg.Precision, s.BatchSize, state, cfg.Version, s); err != nil {
		return
	}
	s.Mutex.Lock()
	defer s.Mutex.Unlock()
	s.FeatureSets[name] = set
	return
}

func (s *_CPUSets) Get(ctx context.Context, name search.SetName) (set search.Set, err error) {
	s.Mutex.Lock()
	set, exist := s.FeatureSets[name]
	if !exist {
		s.Mutex.Unlock()
		err := search.ErrFeatureSetNotFound
		return nil, err
	}
	s.Mutex.Unlock()
	return set, nil
}

func (s *_CPUSets) Delete(ctx context.Context, name search.SetName) {
	s.Mutex.Lock()
	delete(s.FeatureSets, name)
	s.Mutex.Unlock()
}
