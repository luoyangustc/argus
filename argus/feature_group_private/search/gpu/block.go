// +build cublas

package gpu

import (
	"context"
	"sync"

	"github.com/pkg/errors"
	"github.com/unixpickle/cuda"
	"github.com/unixpickle/cuda/cublas"
	"qiniu.com/argus/feature_group_private"
	"qiniu.com/argus/feature_group_private/proto"
	"qiniu.com/argus/feature_group_private/search"
)

type SearchRet struct {
	Ret [][]feature_group.FeatureSearchItem
	Err error
}

type Job struct {
	*cublas.Handle
	Input     cuda.Buffer
	Threshold float32
	Limit     int
	Batch     int
	Result    chan SearchRet
}

type Block struct {
	// block info
	Index     int
	BlockSize int
	Buffer    cuda.Buffer
	Output    cuda.Buffer
	Mutex     sync.RWMutex
	Ctx       *cuda.Context

	// feature info
	Dims      int
	Precision int
	Owner     string
	Empty     []int
	NextIndex int
	IDs       []proto.FeatureID
	QuitCtx   context.Context
	Cancel    context.CancelFunc
	Queue     chan Job
}

func NewBlock(ctx *cuda.Context, index, blockSize int, buffer cuda.Buffer) *Block {
	block := &Block{
		Index:     index,
		BlockSize: blockSize,
		Buffer:    buffer,
		Ctx:       ctx,
	}

	return block
}

func (b *Block) Capacity() int {
	length := b.BlockSize / (b.Precision * b.Dims)
	return len(b.Empty) + (length - b.NextIndex)
}

func (b *Block) Accquire(owner string, dims, precision int) {
	b.Dims = dims
	b.Precision = precision
	b.Owner = owner
	b.IDs = make([]proto.FeatureID, b.BlockSize/(precision*dims))
	b.QuitCtx, b.Cancel = context.WithCancel(context.Background())
	b.Queue = make(chan Job, 100)
	go b.work()
}

func (b *Block) Insert(features []proto.Feature) (err error) {
	if len(features) > b.Capacity() {
		return search.ErrBlockIsFull
	}

	vector, err := ToFloat32Array([]byte(FeautureTOFeatureValue(features)))
	if err != nil {
		return errors.Wrap(err, "ToFloat32Array")
	}
	b.Mutex.Lock()
	defer b.Mutex.Unlock()

	if len(b.Empty) > 0 {
		length := len(b.Empty)
		if len(features) < len(b.Empty) {
			length = len(features)
		}
		for i, index := range b.Empty[:length] {
			buffer := cuda.Slice(b.Buffer, uintptr(index*b.Dims*b.Precision), uintptr((index+1)*b.Dims*b.Precision))
			err = <-b.Ctx.Run(func() (e error) {
				e = cuda.WriteBuffer(buffer, vector[i*b.Dims:(i+1)*b.Dims])
				if e != nil {
					return ErrWriteCudaBuffer
				}
				return
			})
			if err != nil {
				return err
			}
			b.IDs[index] = features[i].ID
		}
	}

	if len(features) > len(b.Empty) {
		buffer := cuda.Slice(b.Buffer, uintptr(b.NextIndex*b.Dims*b.Precision), uintptr((b.NextIndex+len(features)-len(b.Empty))*b.Dims*b.Precision))
		err = <-b.Ctx.Run(func() (e error) {
			e = cuda.WriteBuffer(buffer, vector[len(b.Empty)*b.Dims:])
			if e != nil {
				return ErrWriteCudaBuffer
			}
			return
		})
		if err != nil {
			return err
		}

		for i, feature := range features[len(b.Empty):] {
			b.IDs[b.NextIndex+i] = feature.ID
		}
	}

	if len(features) > len(b.Empty) {
		b.NextIndex += (len(features) - len(b.Empty))
		b.Empty = make([]int, 0)
	} else {
		b.Empty = b.Empty[len(features):]
	}
	return
}

func (b *Block) work() {

	for {
		select {
		case job := <-b.Queue:
			var ret SearchRet
			ret.Ret, ret.Err = b.doSearch(job.Handle, job.Input, job.Batch, job.Threshold, job.Limit)
			job.Result <- ret
		case <-b.QuitCtx.Done():
			return
		}
	}
}

func (b *Block) doSearch(handle *cublas.Handle, inputBuffer cuda.Buffer, batch int, threshold float32, limit int) (ret [][]feature_group.FeatureSearchItem, err error) {
	dimension := b.Dims
	height := b.NextIndex
	if height == 0 {
		ret = make([][]feature_group.FeatureSearchItem, batch)
		return
	}

	b.Mutex.RLock()
	defer b.Mutex.RUnlock()

	vec3 := make([]float32, height*batch)
	err = <-b.Ctx.Run(func() (e error) {
		var alpha, beta float32
		alpha = 1.0
		beta = 0.0
		e = handle.Sgemm(
			cublas.Trans,
			cublas.NoTrans,
			height,
			batch,
			dimension,
			&alpha,
			b.Buffer,
			dimension,
			inputBuffer,
			dimension,
			&beta,
			b.Output,
			height,
		)
		if e != nil {
			return errors.New("fail to sgemm, err:" + e.Error())
		}
		return nil
	})
	if err != nil {
		return
	}
	err = <-b.Ctx.Run(func() (e error) {
		e = cuda.ReadBuffer(vec3, b.Output)
		if e != nil {
			return errors.New("fail to read buffer vec3, err:" + e.Error())
		}
		return nil
	})
	if err != nil {
		return
	}
	for i := 0; i < batch; i++ {
		var result []feature_group.FeatureSearchItem
		indexes, scores := search.MaxNFloat32(vec3[i*height:(i+1)*height], limit)
		for j, index := range indexes {
			if scores[j] < threshold {
				continue
			}
			r := feature_group.FeatureSearchItem{Score: scores[j], ID: b.IDs[index]}
			result = append(result, r)
		}
		ret = append(ret, result)
	}

	return
}

// Search :
//	search N features(s) in the block
//	empty block will return score=0 result
// Inputs :
// 	handle - cublash handle
//	inputbuffer - cuda buffer with  target feature(s)
//	outputbuffer - cuda buffer to store search result temporarily
//	batch - number of target feature(s)
func (b *Block) Search(handle *cublas.Handle, inputBuffer cuda.Buffer, batch int, threshold float32, limit int, retChan chan SearchRet) {
	b.Queue <- Job{
		Handle:    handle,
		Input:     inputBuffer,
		Batch:     batch,
		Threshold: threshold,
		Limit:     limit,
		Result:    retChan,
	}
	return
}

// Delete :
// 	delete N feature(s) from block
// Input :
//	ids - features id(s) to be deleted
// Output:
//	deleted - ids which has been deleted from the block, id(s) of features which are not found in the block will be ignored
//	err - delete error, or nil if success
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
	for id, index := range targets {
		if index != -1 {
			buffer := cuda.Slice(b.Buffer, uintptr(index*b.Dims*b.Precision), uintptr((index+1)*b.Dims*b.Precision))
			err = <-b.Ctx.Run(func() (e error) {
				e = cuda.ClearBuffer(buffer)
				if e != nil {
					return ErrClearCudaBuffer
				}
				return
			})
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

	vector, err := ToFloat32Array([]byte(FeautureTOFeatureValue(features)))
	if err != nil {
		return nil, errors.Wrap(err, "ToFloat32Array")
	}
	b.Mutex.Lock()
	defer b.Mutex.Unlock()
	for id, feature := range features {
		index := targets[feature.ID]
		if index != -1 {
			buffer := cuda.Slice(b.Buffer, uintptr(index*b.Dims*b.Precision), uintptr((index+1)*b.Dims*b.Precision))
			err = <-b.Ctx.Run(func() (e error) {
				e = cuda.WriteBuffer(buffer, vector[id*b.Dims:(id+1)*b.Dims])
				return
			})
			if err != nil {
				return
			}
			updated = append(updated, feature.ID)
		}
	}
	return
}

// Destroy :
// 	destroy the whole block and clear the cuda memory
// Input :
// Output :
// 	err : destroy err, or nil if success
func (b *Block) Destroy() (err error) {
	b.Mutex.Lock()
	defer b.Mutex.Unlock()

	err = <-b.Ctx.Run(func() (e error) {
		e = cuda.ClearBuffer(b.Buffer)
		if e != nil {
			return ErrClearCudaBuffer
		}
		return
	})
	b.Cancel()
	b.Dims = 0
	b.Precision = 0
	b.Owner = ""
	b.IDs = make([]proto.FeatureID, 0)
	b.Empty = make([]int, 0)
	b.NextIndex = 0
	close(b.Queue)

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
	buffer := cuda.Slice(b.Buffer, uintptr(index*b.Dims*b.Precision), uintptr((index+1)*b.Dims*b.Precision))
	err := <-b.Ctx.Run(func() (e error) {
		e = cuda.ReadBuffer(value, buffer)
		if e != nil {
			return errors.New("fail to read buffer from cuda, err:" + e.Error())
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	return proto.FeatureValue(value), nil
}

func (b *Block) GetID(index int) proto.FeatureID {
	if index < 0 || index >= b.NextIndex {
		return proto.FeatureID("")
	}
	return b.IDs[index]
}
