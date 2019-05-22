package cpu

import (
	"sync"

	"qiniu.com/argus/feature_group/distance"
	"qiniu.com/argus/feature_group_private"
	"qiniu.com/argus/feature_group_private/proto"
	"qiniu.com/argus/feature_group_private/search"
)

type Block struct {
	// block info
	Index     int
	BlockSize int
	Buffer    []byte
	Mutex     sync.RWMutex

	// feature info
	Dims      int
	Precision int
	Owner     search.SetName
	Empty     []int
	NextIndex int
	IDs       []proto.FeatureID
}

func NewBlock(index, blockSize int, buffer []byte) *Block {
	return &Block{
		Index:     index,
		BlockSize: blockSize,
		Buffer:    buffer,
	}
}

func (b *Block) Capacity() int {
	length := b.BlockSize / (b.Precision * b.Dims)
	return len(b.Empty) + (length - b.NextIndex)
}

func (b *Block) Accquire(owner search.SetName, dims, precision int) {
	b.Dims = dims
	b.Precision = precision
	b.Owner = owner
	b.IDs = make([]proto.FeatureID, b.BlockSize/(precision*dims))
}

func (b *Block) Insert(features []proto.Feature) (err error) {
	if len(features) > b.Capacity() {
		return search.ErrBlockIsFull
	}

	{
		b.Mutex.Lock()
		defer b.Mutex.Unlock()

		if len(b.Empty) > 0 {
			length := len(b.Empty)
			if len(features) < len(b.Empty) {
				length = len(features)
			}
			for i, index := range b.Empty[:length] {
				copy(b.Buffer[index*b.Dims*b.Precision:(index+1)*b.Dims*b.Precision], features[i].Value)
				b.IDs[index] = features[i].ID
			}
		}

		if len(features) > len(b.Empty) {
			buffer := b.Buffer[b.NextIndex*b.Dims*b.Precision : (b.NextIndex+len(features)-len(b.Empty))*b.Dims*b.Precision]
			for i, feature := range features[len(b.Empty):] {
				copy(buffer[i*b.Dims*b.Precision:(i+1)*b.Dims*b.Precision], feature.Value)
				b.IDs[b.NextIndex+i] = feature.ID
			}
		}

		if len(features) > len(b.Empty) {
			b.NextIndex += (len(features) - len(b.Empty))
			b.Empty = make([]int, 0)
		} else {
			b.Empty = b.Empty[len(features):]
		}
	}
	return
}

func (b *Block) Search(features []proto.FeatureValue, limit int) (ret [][]feature_group.FeatureSearchItem, err error) {
	b.Mutex.RLock()
	defer b.Mutex.RUnlock()
	scores := make([]float32, b.NextIndex)
	for _, feature := range features {
		var (
			score   []float32
			result  []feature_group.FeatureSearchItem
			indexes []int
		)
		/*
			for i := 0; i < b.NextIndex; i++ {
				score := distanceCosineCgo(b.Buffer[i*b.Dims*b.Precision:(i+1)*b.Dims*b.Precision], feature)
				scores = append(scores, score)
			}
		*/
		distance.DistancesCosineCgoFlat(feature, b.Buffer[:b.NextIndex*b.Precision*b.Dims], scores)
		indexes, score = search.MaxNFloat32(scores, limit)
		for j, index := range indexes {
			r := feature_group.FeatureSearchItem{Score: score[j], ID: b.IDs[index]}
			result = append(result, r)
		}
		ret = append(ret, result)
	}

	return
}

func (b *Block) Delete(ids []proto.FeatureID) (deleted []proto.FeatureID, err error) {
	targets := make(map[proto.FeatureID]int, 0)
	for _, id := range ids {
		targets[id] = -1
	}
	for index, value := range b.IDs {
		if _, exist := targets[value]; exist {
			targets[value] = index
		}
	}
	if len(targets) == 0 {
		return
	}
	b.Mutex.Lock()
	defer b.Mutex.Unlock()
	empty := make([]byte, b.Dims*b.Precision)
	for id, index := range targets {
		if index != -1 {
			copy(b.Buffer[index*b.Dims*b.Precision:(index+1)*b.Dims*b.Precision], empty)
			b.IDs[index] = ""
			b.Empty = append(b.Empty, index)
			deleted = append(deleted, id)
		}
	}
	return
}

// Update :
// 	update N feature(s) in the block
func (b *Block) Update(features []proto.Feature) (updated []proto.FeatureID, err error) {
	targets := make(map[proto.FeatureID]int, 0)
	for _, feature := range features {
		targets[feature.ID] = -1
	}
	for index, value := range b.IDs {
		if _, exist := targets[value]; exist {
			targets[value] = index
		}
	}
	if len(targets) == 0 {
		return
	}

	b.Mutex.Lock()
	defer b.Mutex.Unlock()
	for id, feature := range features {
		index := targets[feature.ID]
		if index != -1 {
			buffer := b.Buffer[index*b.Dims*b.Precision : (index+1)*b.Dims*b.Precision]
			copy(buffer, features[id].Value)
			updated = append(updated, feature.ID)
		}
	}
	return
}

func (b *Block) Destroy() (err error) {
	b.Mutex.Lock()
	defer b.Mutex.Unlock()

	// empty := make([]byte, b.BlockSize)
	// copy(b.Buffer, empty)
	for i := 0; i < b.BlockSize; i++ {
		b.Buffer[i] = 0
	}

	b.Dims = 0
	b.Precision = 0
	b.Owner = ""
	b.IDs = make([]proto.FeatureID, 0)
	b.Empty = make([]int, 0)
	b.NextIndex = 0

	return
}

func (b *Block) Get(target proto.FeatureID) (proto.FeatureValue, error) {
	index := -1
	for i, id := range b.IDs {
		if id == target {
			index = i
		}
	}
	if index == -1 {
		return nil, nil
	}
	value := make([]byte, b.Dims*b.Precision)
	copy(value, b.Buffer[index*b.Dims*b.Precision:(index+1)*b.Dims*b.Precision])
	return proto.FeatureValue(value), nil
}

func (b *Block) Features() (values []proto.FeatureValue, ids []proto.FeatureID) {
	for index := 0; index < b.NextIndex; index++ {
		if len(b.IDs[index]) > 0 {
			values = append(values, b.Buffer[index*b.Precision*b.Dims:(index+1)*b.Precision*b.Dims])
			ids = append(ids, b.IDs[index])
		}
	}
	return
}
