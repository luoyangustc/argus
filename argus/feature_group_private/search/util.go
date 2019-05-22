package search

import (
	"container/heap"
	"encoding/binary"
	"math"
	"math/rand"
	"sort"
	"time"

	"qiniu.com/argus/feature_group_private"
	"qiniu.com/argus/feature_group_private/proto"
)

func Feature2DTo1D(feautures []proto.Feature) (ret proto.FeatureValue) {
	for _, feature := range feautures {
		ret = append(ret, feature.Value...)
	}
	return
}

func FeatureToFloat32(feature proto.Feature) (ret []float32, err error) {
	length := len(feature.Value)
	if length%4 != 0 {
		err = ErrInvalidFeatureValue
		return
	}

	for i := 0; i < length; i = i + 4 {
		value := math.Float32frombits(binary.LittleEndian.Uint32(feature.Value[i*4 : (i+1)*4]))
		ret = append(ret, value)
	}
	return
}

type _result struct {
	Value float32
	Index int
}

type topNHeap []_result

func (h topNHeap) Len() int            { return len(h) }
func (h topNHeap) Less(i, j int) bool  { return h[i].Value < h[j].Value }
func (h topNHeap) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *topNHeap) Push(x interface{}) { *h = append(*h, x.(_result)) }
func (h *topNHeap) Pop() interface{} {
	x := (*h)[len(*h)-1]
	*h = (*h)[:len(*h)-1]
	return x
}

func MaxNFloat32(vector []float32, limit int) ([]int, []float32) {
	var (
		scores = &topNHeap{}
		result _result
	)
	result.Value, result.Index = -9999, -1
	for i := 0; i < limit; i++ {
		heap.Push(scores, result)
	}
	heap.Init(scores)
	for k, v := range vector {
		if v > (*scores)[0].Value {
			heap.Pop(scores)
			result.Value, result.Index = v, k
			heap.Push(scores, result)
		}
	}
	index := make([]int, limit)
	max := make([]float32, limit)
	for i := 0; i < limit; i++ {
		result = heap.Pop(scores).(_result)
		index[limit-1-i] = result.Index
		max[limit-1-i] = result.Value
	}
	if len(vector) < limit {
		index = index[:len(vector)]
		max = max[:len(vector)]
	}
	return index, max
}

func MaxNFeatureSearchResult(vector []feature_group.FeatureSearchItem, limit int) ([]int, []feature_group.FeatureSearchItem) {
	type _result struct {
		Value feature_group.FeatureSearchItem
		Index int
	}
	var scores []_result
	for k, v := range vector {
		scores = append(scores, _result{Value: v, Index: k})
	}
	sort.Slice(scores, func(i, j int) bool { return scores[i].Value.Score > scores[j].Value.Score })
	var index []int
	var max []feature_group.FeatureSearchItem
	for i := 0; i < limit && i < len(scores); i++ {
		index = append(index, scores[i].Index)
		max = append(max, scores[i].Value)
	}
	return index, max
}

func GetRandomString(length int) string {
	str := "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_-"
	bytes := []byte(str)
	result := []byte{}
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	for i := 0; i < length; i++ {
		result = append(result, bytes[r.Intn(len(str))])
	}
	return string(result)
}
