package search

import (
	"container/heap"
	"sort"

	"qiniu.com/argus/tuso/proto"
)

const (
	// MaxAvailableFeatures is the maximum number of features in Result()
	MaxAvailableFeatures = 256 * 1024
)

type Searcher interface {
	Append(item proto.FeatureItem)
	AppendDistanceItem(distanceItem proto.DistanceItem)
	Result() []proto.DistanceItem
	SortedResult() []proto.DistanceItem
}

func NewTopNSearcher(feature proto.Feature, n int) Searcher {
	searcher := &topNSearcher{
		feature:  feature,
		n:        n,
		distance: BestCosineDistance().f,
	}

	// pre-alloc the memory in this case
	searcher.result = make(topNHeap, 0, searcher.n)
	heap.Init(&searcher.result)

	return searcher
}

func NewThresholdSearcher(feature proto.Feature, threshold float32, max int) Searcher {
	searcher := &thresholdSearcher{
		feature:   feature,
		threshold: threshold,
		max:       max,
		distance:  BestCosineDistance().f,
	}
	if searcher.max <= 0 {
		searcher.max = MaxAvailableFeatures
	}
	if searcher.max > MaxAvailableFeatures {
		searcher.max = MaxAvailableFeatures
	}

	return searcher
}

type topNSearcher struct {
	feature  proto.Feature
	n        int
	result   topNHeap
	distance func(feature1, feature2 proto.Feature) float32
}

func (searcher *topNSearcher) Append(item proto.FeatureItem) {
	distanceItem := proto.DistanceItem{
		Index:    item.Index,
		Offset:   item.Offset,
		Distance: searcher.distance(item.Feature, searcher.feature),
	}

	if searcher.result.Len() < searcher.n {
		heap.Push(&searcher.result, distanceItem)
	} else {
		heap.Push(&searcher.result, distanceItem)
		heap.Pop(&searcher.result)
	}
}

func (searcher *topNSearcher) AppendDistanceItem(distanceItem proto.DistanceItem) {
	if searcher.result.Len() < searcher.n {
		heap.Push(&searcher.result, distanceItem)
	} else {
		heap.Push(&searcher.result, distanceItem)
		heap.Pop(&searcher.result)
	}
}

func (searcher *topNSearcher) Result() []proto.DistanceItem {
	return searcher.result
}

func (searcher *topNSearcher) SortedResult() []proto.DistanceItem {
	// sort by distance asce
	sort.Sort(sort.Reverse(searcher.result))
	return searcher.result
}

type thresholdSearcher struct {
	feature   proto.Feature
	threshold float32
	max       int
	result    []proto.DistanceItem
	distance  func(feature1, feature2 proto.Feature) float32
}

func (searcher *thresholdSearcher) Append(item proto.FeatureItem) {
	if len(searcher.result) < searcher.max {
		distanceItem := proto.DistanceItem{
			Index:    item.Index,
			Offset:   item.Offset,
			Distance: searcher.distance(item.Feature, searcher.feature),
		}
		if distanceItem.Distance > searcher.threshold {
			searcher.result = append(searcher.result, distanceItem)
		}
	} else {
		// discard
	}
}

func (searcher *thresholdSearcher) AppendDistanceItem(distanceItem proto.DistanceItem) {
	if len(searcher.result) < searcher.max {
		if distanceItem.Distance > searcher.threshold {
			searcher.result = append(searcher.result, distanceItem)
		}
	} else {
		// discard
	}
}

func (searcher *thresholdSearcher) Result() []proto.DistanceItem {
	return searcher.result
}

func (searcher *thresholdSearcher) SortedResult() []proto.DistanceItem {
	// sort by distance asce
	sort.Slice(searcher.result, func(i, j int) bool {
		return searcher.result[i].Distance > searcher.result[j].Distance
	})
	return searcher.result
}

// A maximum heap implementation for proto.DistanceItem
type topNHeap []proto.DistanceItem

func (h topNHeap) Len() int {
	return len(h)
}
func (h topNHeap) Swap(i, j int) {
	h[i], h[j] = h[j], h[i]
}
func (h topNHeap) Less(i, j int) bool {
	return h[i].Distance < h[j].Distance
}

func (h *topNHeap) Push(x interface{}) {
	item := x.(proto.DistanceItem)
	*h = append(*h, item)
}

func (h *topNHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}
