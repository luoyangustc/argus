// +build cublas

package gpu

import (
	"context"
	"net/http"
	"sync"
	"sync/atomic"

	"github.com/pkg/errors"
	"github.com/unixpickle/cuda"
	"github.com/unixpickle/cuda/cublas"
	"qbox.us/net/httputil"
	"qiniu.com/argus/feature_group_private"
	"qiniu.com/argus/feature_group_private/proto"
	"qiniu.com/argus/feature_group_private/search"
)

// set working state
const (
	SetStateUnknown search.SetState = iota
	SetStateCreated
	SetStateInitialized
)

//-------------------------- Sets --------------------------//

var _ search.Sets = new(Sets)
var _ search.Set = new(Set)

type Sets struct {
	search.Config
	sets    map[search.SetName]*Set
	Cache   *Cache
	Context *cuda.Context
	Mutex   sync.Mutex
}

func NewSets(config search.Config) (*Sets, error) {
	s := &Sets{
		Config: config,
		sets:   make(map[search.SetName]*Set, 0),
	}

	devices, err := cuda.AllDevices()
	if err != nil {
		return nil, ErrAllDevice
	}
	if config.DeviceID < 0 || config.DeviceID > len(devices) {
		return nil, search.ErrInvalidDeviceID
	}

	if s.Context, err = cuda.NewContext(devices[config.DeviceID], -1); err != nil {
		return nil, ErrCreateContext
	}

	if s.Cache, err = NewCache(s.Context, config.BlockNum, config.BlockSize, config.BatchSize, config.Dimension); err != nil {
		return nil, err
	}
	return s, nil
}

func (s *Sets) New(ctx context.Context, name search.SetName, config search.Config, state search.SetState) (err error) {
	s.Mutex.Lock()
	if _, exist := s.sets[name]; exist {
		s.Mutex.Unlock()
		err = search.ErrFeatureSetExist
		return
	}
	s.Mutex.Unlock()

	blockNum := config.Capacity / (s.BlockSize / (config.Dimension * config.Precision))
	if _, err = s.Cache.GetEmptyBlock(blockNum); err != nil {
		err = httputil.NewError(http.StatusBadRequest, err.Error())
		return
	}

	set := &Set{
		Dimension:       config.Dimension,
		Precision:       config.Precision,
		Version:         config.Version,
		Ctx:             s.Cache.Ctx,
		BlockFeatureNum: s.Cache.BlockSize / (config.Dimension * config.Precision),
		Name:            name,
		Batch:           s.BatchSize,
		Cache:           s.Cache,
		State:           uint32(state),
		manager:         s,
	}
	set.Handle, err = cublas.NewHandle(s.Cache.Ctx)
	if err != nil {
		return
	}

	s.Mutex.Lock()
	defer s.Mutex.Unlock()
	s.sets[name] = set
	return
}

func (s *Sets) Get(ctx context.Context, name search.SetName) (search.Set, error) {
	s.Mutex.Lock()
	set, exist := s.sets[name]
	if !exist {
		s.Mutex.Unlock()
		err := search.ErrFeatureSetNotFound
		return nil, err
	}
	s.Mutex.Unlock()
	return set, nil
}

func (s *Sets) Delete(ctx context.Context, name search.SetName) {
	s.Mutex.Lock()
	delete(s.sets, name)
	s.Mutex.Unlock()
}

//-------------------------- Set --------------------------//
type Set struct {
	Ctx             *cuda.Context
	Name            search.SetName
	Dimension       int
	BlockFeatureNum int
	Precision       int
	Batch           int
	Version         uint64
	State           uint32
	Blocks          []*Block
	Cache           *Cache
	Handle          *cublas.Handle
	SearchLock      sync.Mutex
	manager         *Sets
}

func (s *Set) Config(ctx context.Context) search.Config {
	return search.Config{
		Dimension: s.Dimension,
		Precision: s.Precision,
		Version:   s.Version,
	}
}

func (s *Set) SetState(ctx context.Context, state search.SetState) (err error) {
	if state < SetStateUnknown || state > SetStateInitialized {
		err = search.ErrInvalidSetState
		return
	}
	atomic.StoreUint32(&s.State, uint32(state))
	return
}

func (s *Set) Destroy(ctx context.Context) (err error) {
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

func (s *Set) Add(ctx context.Context, features ...proto.Feature) (err error) {
	var empty int

	for _, block := range s.Blocks {
		empty += block.Capacity()
	}

	if len(features) > empty {
		remain := len(features) - empty
		blockLength := s.manager.BlockSize / (s.Dimension * s.Precision)
		blockNum := (remain + blockLength - 1) / blockLength
		var blocks []*Block
		if blocks, err = s.Cache.GetEmptyBlock(blockNum); err != nil {
			return
		}
		for _, block := range blocks {
			block.Accquire(string(s.Name), s.Dimension, s.Precision)
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

func (s *Set) Delete(ctx context.Context, ids ...proto.FeatureID) (deleted []proto.FeatureID, err error) {
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

func (s *Set) Update(ctx context.Context, features ...proto.Feature) (err error) {
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
	if atomic.LoadUint32(&s.State) == uint32(SetStateInitialized) {
		atomic.AddUint64(&s.Version, 1)
	}
	return
}

func (s *Set) Search(ctx context.Context, threshold float32, limit int, features ...proto.FeatureValue) (ret [][]feature_group.FeatureSearchItem, err error) {

	batch := len(features)
	if batch > s.Batch {
		batch = s.Batch
	}
	ffs := features
	inputBuffer, e := s.Cache.AccquireBuffer(batch * s.Dimension * s.Precision)
	if err != nil {
		return nil, e
	}
	for len(ffs) > 0 {
		if len(ffs) > batch {
			features = ffs[:batch]
			ffs = ffs[batch:]
		} else {
			features = ffs
			ffs = ffs[len(ffs):]
		}

		f := make([]float32, 0)
		for _, v := range features {
			r, err := ToFloat32Array(v)
			if err != nil {
				return nil, errors.Wrap(err, "ToFloat32Array")
			}
			f = append(f, r...)
		}

		err = <-s.Ctx.Run(func() (e error) {
			e = cuda.WriteBuffer(inputBuffer, f)
			if e != nil {
				return errors.New("fail to write input buffer target, err:" + err.Error())
			}
			return
		})
		if err != nil {
			return nil, search.ErrWriteInputBuffer
		}

		results := make([][]feature_group.FeatureSearchItem, len(features))
		retChan := make(chan SearchRet, len(s.Blocks))
		for _, block := range s.Blocks {
			block.Search(s.Handle, inputBuffer, len(features), threshold, limit, retChan)
		}
		for _, _ = range s.Blocks {
			result := <-retChan
			err = result.Err
			if result.Err == nil {
				for b, r := range result.Ret {
					results[b] = append(results[b], r...)
				}
			}
		}
		close(retChan)
		if err != nil {
			return nil, err
		}
		for _, result := range results {
			_, features := MaxNFeatureResult(result, limit)
			ret = append(ret, features)
		}
	}

	return

}

func (s *Set) SpaceAvailable(ctx context.Context, size int) bool {
	var empty int
	for _, block := range s.Blocks {
		empty += block.Capacity()
	}
	return empty+s.BlockFeatureNum*len(s.Cache.AvaiableBlocks()) >= size
}

func (s *Set) Get(ctx context.Context, id proto.FeatureID) (value proto.FeatureValue, err error) {

	for _, block := range s.Blocks {
		value, err = block.Get(id)
		if err != nil || value != nil {
			return
		}
	}
	return nil, search.ErrFeatureNotFound
}

func (s *Set) Compare(ctx context.Context,
	threshold float32, limit int,
	target search.Set,
) (
	[]feature_group.FeatureCompareItem,
	error,
) {

	targetSet, ok := target.(*Set)
	if !ok {
		return nil, search.ErrInvalidCompareTargetSet
	}

	var (
		ret []feature_group.FeatureCompareItem
		err error
	)
	batch := s.Batch
	for _, targetBlock := range targetSet.Blocks {
		targetBlock.Mutex.RLock()
		retChans := make([]chan SearchRet, len(s.Blocks))
		for i, _ := range retChans {
			retChans[i] = make(chan SearchRet, (targetBlock.NextIndex+batch-1)/batch)
		}
		go func() {
			for index := 0; index < targetBlock.NextIndex; index += batch {
				len := batch
				if batch > targetBlock.NextIndex-index {
					len = targetBlock.NextIndex - index
				}
				inputBuffer := cuda.Slice(targetBlock.Buffer, uintptr(index*targetBlock.Precision*targetBlock.Dims), uintptr((index+len)*targetBlock.Precision*targetBlock.Dims))
				for i, block := range s.Blocks {
					block.Search(s.Handle, inputBuffer, len, threshold, limit, retChans[i])
				}
			}
		}()
		for i := 0; i < (targetBlock.NextIndex+batch-1)/batch; i++ {
			results := make([][]feature_group.FeatureSearchItem, batch)
			for j, _ := range s.Blocks {
				result := <-retChans[j]
				err = result.Err
				if result.Err == nil {
					for b, r := range result.Ret {
						results[b] = append(results[b], r...)
					}
				}
			}
			for j, result := range results {
				if id := targetBlock.GetID(i*batch + j); len(id) > 0 {
					_, features := MaxNFeatureResult(result, limit)
					ret = append(ret, feature_group.FeatureCompareItem{
						ID:    id,
						Faces: features,
					})
				}
			}
		}
		for i, _ := range s.Blocks {
			close(retChans[i])
		}

		targetBlock.Mutex.RUnlock()
		if err != nil {
			return nil, err
		}
	}
	return ret, nil
}
