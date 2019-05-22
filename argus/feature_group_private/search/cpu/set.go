package cpu

import (
	"context"
	"runtime"
	"sync"
	"sync/atomic"

	"qiniu.com/argus/feature_group_private"
	"qiniu.com/argus/feature_group_private/proto"
	"qiniu.com/argus/feature_group_private/search"
)

type FeatureSet struct {
	Name            search.SetName
	Dimension       int
	BlockFeatureNum int
	Precision       int
	Batch           int
	Version         uint64
	State           uint32
	Blocks          []*Block
	Cache           *Cache
	SearchLock      sync.RWMutex
	manager         search.Sets
}

var _ search.Set = &FeatureSet{}

// set working state
const (
	SetStateUnknown search.SetState = iota
	SetStateCreated
	SetStateInitialized
)

func NewSet(cache *Cache, name search.SetName, dims, precision, batch int, state search.SetState, version uint64, sets search.Sets) (set *FeatureSet, err error) {
	set = &FeatureSet{
		Dimension:       dims,
		Precision:       precision,
		BlockFeatureNum: cache.BlockSize / (dims * precision),
		Name:            name,
		Batch:           batch,
		Cache:           cache,
		Version:         version,
		State:           uint32(state),
		manager:         sets,
	}
	return
}

func (s *FeatureSet) Config(ctx context.Context) search.Config {
	return search.Config{
		Dimension: s.Dimension,
		Precision: s.Precision,
		Version:   s.Version,
	}
}

func (s *FeatureSet) SetState(ctx context.Context, state search.SetState) (err error) {
	atomic.StoreUint32(&s.State, uint32(state))
	return
}

func (s *FeatureSet) Add(ctx context.Context, features ...proto.Feature) (err error) {
	var empty int

	for _, block := range s.Blocks {
		empty += block.Capacity()
	}

	if len(features) > empty {
		remain := len(features) - empty
		blockLength := s.Cache.BlockSize / (s.Dimension * s.Precision)
		blockNum := (remain + blockLength - 1) / blockLength
		var blocks []*Block
		if blocks, err = s.Cache.GetEmptyBlock(blockNum); err != nil {
			return
		}
		for _, block := range blocks {
			block.Accquire(s.Name, s.Dimension, s.Precision)
		}
		s.Blocks = append(s.Blocks, blocks...)
	}
	offset := 0
	remain := len(features)
	for _, block := range s.Blocks {
		length := block.Capacity()
		if length > remain {
			length = remain
		}
		if length > 0 {
			block.Insert(features[offset:(offset + length)])
			offset += length
			remain -= length
		}
		if remain <= 0 {
			break
		}
	}
	if atomic.LoadUint32(&s.State) == uint32(SetStateInitialized) {
		atomic.AddUint64(&s.Version, 1)
	}
	return
}

func (s *FeatureSet) Search(ctx context.Context, threshold float32, limit int, features ...proto.FeatureValue) (ret [][]feature_group.FeatureSearchItem, err error) {
	s.SearchLock.RLock()
	defer s.SearchLock.RUnlock()

	wg := sync.WaitGroup{}
	type output struct {
		err    error
		index  int
		result [][]feature_group.FeatureSearchItem
	}
	results := make([][]feature_group.FeatureSearchItem, len(features))
	inputChan := make(chan int, len(s.Blocks))
	outputChan := make(chan output, len(s.Blocks))
	for i := 0; i < runtime.NumCPU(); i++ {
		wg.Add(1)
		go func(id int) {
			for index := range inputChan {
				var (
					r output = output{index: index}
				)
				ffs := features
				for len(ffs) > 0 {
					var (
						result   [][]feature_group.FeatureSearchItem
						toSearch []proto.FeatureValue
						e        error
					)
					if len(ffs) > s.Batch {
						toSearch = ffs[:s.Batch]
						ffs = ffs[s.Batch:]
					} else {
						toSearch = ffs
						ffs = ffs[len(ffs):]
					}
					if result, e = s.Blocks[index].Search(toSearch, limit); e != nil {
						r.err = e
						break
					}
					r.result = append(r.result, result...)
				}
				outputChan <- r
			}
			defer wg.Done()
		}(i)
	}
	for i := 0; i < len(s.Blocks); i++ {
		inputChan <- i
	}
	close(inputChan)
	wg.Wait()
	close(outputChan)
	for s := range outputChan {
		if s.err != nil {
			return nil, err
		}
		for b, r := range s.result {
			results[b] = append(results[b], r...)
		}
	}
	for _, result := range results {
		_, features := search.MaxNFeatureSearchResult(result, limit)
		var filterd []feature_group.FeatureSearchItem
		for _, feature := range features {
			if feature.Score >= threshold {
				filterd = append(filterd, feature)
			}
		}
		ret = append(ret, filterd)
	}
	return
}

func (s *FeatureSet) Delete(ctx context.Context, ids ...proto.FeatureID) (deleted []proto.FeatureID, err error) {
	var del []proto.FeatureID

	// can not delete empty id in the set
	for _, id := range ids {
		if len(id) == 0 {
			err = search.ErrDeleteEmptyID
			return
		}
	}

	for _, block := range s.Blocks {
		if len(ids) == 0 {
			break
		}
		del, err = block.Delete(ids)
		if err != nil {
			return
		}
		if len(del) == 0 {
			continue
		}
		var remain []proto.FeatureID
		remain = append(remain, ids...)
		ids = make([]proto.FeatureID, 0)
		deleted = append(deleted, del...)
		for _, id := range remain {
			flag := false
			for _, d := range del {
				if id == d {
					flag = true
					break
				}
			}
			if !flag {
				ids = append(ids, id)
			}
		}
	}
	if atomic.LoadUint32(&s.State) == uint32(SetStateInitialized) {
		atomic.AddUint64(&s.Version, 1)
	}
	return
}

func (s *FeatureSet) Update(ctx context.Context, features ...proto.Feature) (err error) {
	if atomic.LoadUint32(&s.State) == uint32(SetStateInitialized) {
		atomic.AddUint64(&s.Version, 1)
	}

	var updated []proto.FeatureID
	for _, block := range s.Blocks {
		if len(features) == 0 {
			break
		}
		updated, err = block.Update(features)
		if err != nil {
			return
		}
		if len(updated) == 0 {
			continue
		}
		var remain []proto.Feature
		remain = append(remain, features...)
		features = make([]proto.Feature, 0)
		for _, feature := range remain {
			flag := false
			for _, id := range updated {
				if feature.ID == id {
					flag = true
					break
				}
			}
			if !flag {
				features = append(features, feature)
			}
		}
	}
	return
}

func (s *FeatureSet) Destroy(ctx context.Context) (err error) {
	s.manager.Delete(ctx, s.Name)
	s.SearchLock.Lock()
	defer s.SearchLock.Unlock()

	for _, block := range s.Blocks {
		if err = block.Destroy(); err != nil {
			return
		}
	}
	return
}

func (s *FeatureSet) SpaceAvailable(ctx context.Context, size int) bool {
	var empty int
	for _, block := range s.Blocks {
		empty += block.Capacity()
	}
	return empty+s.BlockFeatureNum*len(s.Cache.AvaiableBlocks()) >= size
}

func (s *FeatureSet) Get(ctx context.Context, id proto.FeatureID) (value proto.FeatureValue, err error) {
	for _, block := range s.Blocks {
		value, err = block.Get(id)
		if err != nil || value != nil {
			return
		}
	}
	return nil, search.ErrFeatureNotFound
}

func (s *FeatureSet) Compare(ctx context.Context, threshold float32, limit int, target search.Set) ([]feature_group.FeatureCompareItem, error) {
	targetSet, ok := target.(*FeatureSet)
	if !ok {
		return nil, search.ErrInvalidCompareTargetSet
	}

	var (
		ret []feature_group.FeatureCompareItem
	)

	for _, targetBlock := range targetSet.Blocks {
		targetBlock.Mutex.RLock()
		features, ids := targetBlock.Features()

		results, err := s.Search(ctx, threshold, limit, features...)
		if err != nil {
			targetBlock.Mutex.RUnlock()
			return nil, err
		}

		for index, result := range results {
			if len(result) > 0 {
				ret = append(ret, feature_group.FeatureCompareItem{
					ID:    ids[index],
					Faces: result,
				})
			}
		}
		targetBlock.Mutex.RUnlock()
	}

	return ret, nil
}
