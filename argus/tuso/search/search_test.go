package search

import (
	"math/rand"
	"sort"
	"testing"

	"qiniu.com/argus/tuso/proto"
	"qiniu.com/argus/tuso/utils"

	"github.com/stretchr/testify/assert"
)

const (
	FeatureSize          = 16 * 1024
	FeatureSizeInFloat32 = FeatureSize / 4
	FeatureSetSize       = 1 * 1024 * 1024 * 1024
	FeaturesPerSet       = FeatureSetSize / FeatureSize
)

type FeatureItemFloat32 struct {
	Feature []float32
	Index   int
	Offset  int
}

func toFeatureItem(item FeatureItemFloat32) proto.FeatureItem {
	featureItem := proto.FeatureItem{
		Index:  item.Index,
		Offset: item.Offset,
	}
	featureItem.Feature = make(proto.Feature, 0, FeatureSize)
	for _, value := range item.Feature {
		featureItem.Feature = append(featureItem.Feature, utils.Float32ToByte(value)...)
	}
	NormFeatures(featureItem.Feature, FeatureSize)

	return featureItem
}

func prepareFeatureItems(count int) []proto.FeatureItem {
	featureItemsFloat32 := make([]FeatureItemFloat32, count)
	for index := range featureItemsFloat32 {
		featureItemsFloat32[index] = FeatureItemFloat32{
			Index:  rand.Int() % 10,
			Offset: rand.Int() % count,
		}
		featureItemsFloat32[index].Feature = make([]float32, FeatureSizeInFloat32)
		for i := range featureItemsFloat32[index].Feature {
			featureItemsFloat32[index].Feature[i] = rand.Float32()
		}
	}

	featureItems := make([]proto.FeatureItem, count)
	for index, item := range featureItemsFloat32 {
		featureItems[index] = toFeatureItem(item)
	}

	return featureItems
}

func float32tobytes(floats ...float32) []byte {
	ret := make([]byte, 0, len(floats)*4)
	for _, value := range floats {
		ret = append(ret, utils.Float32ToByte(value)...)
	}
	return ret
}

func TestTopN(t *testing.T) {
	assert := assert.New(t)

	// generate the features
	count := FeaturesPerSet / 64
	featureItems := prepareFeatureItems(count)

	// randomly select one to search
	targetItem := featureItems[rand.Int()%count]
	t.Logf("[Target]Index: %d, Offset: %d\n", targetItem.Index, targetItem.Offset)

	// topN
	topN := 10
	topNFeatureItems := make([]proto.FeatureItem, len(featureItems))
	copy(topNFeatureItems, featureItems)
	topNSearcher := NewTopNSearcher(targetItem.Feature, topN)
	for _, item := range topNFeatureItems {
		topNSearcher.Append(item)
	}
	topNResult := topNSearcher.SortedResult()
	for index, item := range topNResult {
		t.Logf("[%dth]Index: %d, Offset: %d, Distance: %f\n", index, item.Index, item.Offset, item.Distance)
	}

	// sort
	sortFeatureItems := make([]proto.FeatureItem, len(featureItems))
	copy(sortFeatureItems, featureItems)
	sort.Slice(sortFeatureItems, func(i, j int) bool {
		distanceWithI := BestCosineDistance().f(targetItem.Feature, sortFeatureItems[i].Feature)
		distanceWithJ := BestCosineDistance().f(targetItem.Feature, sortFeatureItems[j].Feature)

		return distanceWithI > distanceWithJ
	})
	// 余弦距离越大代表越接近
	assert.True(BestCosineDistance().f(sortFeatureItems[0].Feature, targetItem.Feature) > BestCosineDistance().f(sortFeatureItems[1].Feature, targetItem.Feature))
	// 第一个就是 targetItem ，余弦聚类应该是1
	assert.InDelta(BestCosineDistance().f(sortFeatureItems[0].Feature, targetItem.Feature), 1, 0.000001)
	// compare topN with sort
	assert.Equal(len(topNResult), topN)
	for index, item := range topNResult {
		assert.Equal(item.Index, sortFeatureItems[index].Index)
		assert.Equal(item.Offset, sortFeatureItems[index].Offset)
	}
}

func TestThreshold(t *testing.T) {
	assert := assert.New(t)

	// generate the features
	count := FeaturesPerSet / 64
	featureItems := prepareFeatureItems(count)

	// randomly select one to search
	targetItem := featureItems[rand.Int()%count]
	t.Logf("[Target]Index: %d, Offset: %d\n", targetItem.Index, targetItem.Offset)

	// threshold
	var threshold float32 = 0.7
	thresholdSearcher := NewThresholdSearcher(targetItem.Feature, threshold, 0)
	for _, item := range featureItems {
		thresholdSearcher.Append(item)
	}
	thresholdResult := thresholdSearcher.SortedResult()
	for index, item := range thresholdResult {
		t.Logf("[%dth]Index: %d, Offset: %d, Distance: %f\n", index, item.Index, item.Offset, item.Distance)
	}

	assert.Equal(thresholdResult[0].Index, targetItem.Index)
	assert.Equal(thresholdResult[0].Offset, targetItem.Offset)

	// sort
	sortFeatureItems := make([]proto.FeatureItem, len(featureItems))
	copy(sortFeatureItems, featureItems)
	sort.Slice(sortFeatureItems, func(i, j int) bool {
		distanceWithI := BestCosineDistance().f(targetItem.Feature, sortFeatureItems[i].Feature)
		distanceWithJ := BestCosineDistance().f(targetItem.Feature, sortFeatureItems[j].Feature)

		return distanceWithI > distanceWithJ
	})
	// 余弦距离越大代表越接近
	assert.True(BestCosineDistance().f(sortFeatureItems[0].Feature, targetItem.Feature) > BestCosineDistance().f(sortFeatureItems[1].Feature, targetItem.Feature))

	for index, item := range thresholdResult {
		assert.Equal(item.Index, sortFeatureItems[index].Index)
		assert.Equal(item.Offset, sortFeatureItems[index].Offset)
		assert.True(item.Distance > threshold)
	}

}

func BenchmarkSort(b *testing.B) {
	// generate the features
	featureItems := prepareFeatureItems(FeaturesPerSet)

	targetItem := featureItems[rand.Int()%FeaturesPerSet]

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		distanceItems := make([]proto.DistanceItem, FeaturesPerSet)
		for index, featureItem := range featureItems {
			distanceItems[index].Index = featureItem.Index
			distanceItems[index].Offset = featureItem.Offset
			distanceItems[index].Distance = BestCosineDistance().f(targetItem.Feature, featureItem.Feature)
		}

		sort.Slice(distanceItems, func(i, j int) bool {
			return distanceItems[i].Distance < distanceItems[j].Distance
		})

		b.SetBytes(int64(len(featureItems) * FeatureSize))
	}
}

func BenchmarkTopN(b *testing.B) {
	// generate the features
	featureItems := prepareFeatureItems(FeaturesPerSet)

	topN := 10
	targetItem := featureItems[rand.Int()%FeaturesPerSet]

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		topNSearcher := NewTopNSearcher(targetItem.Feature, topN)
		for _, item := range featureItems {
			topNSearcher.Append(item)
		}
		var _ = topNSearcher.Result()

		b.SetBytes(int64(len(featureItems) * FeatureSize))
	}
}
