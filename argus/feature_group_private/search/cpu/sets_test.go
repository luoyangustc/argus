package cpu

import (
	"context"
	"encoding/binary"
	"math"
	"math/rand"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"qiniu.com/argus/feature_group_private/proto"
	"qiniu.com/argus/feature_group_private/search"
)

const (
	blockFeatureCount = 10000
)

var (
	sets *_CPUSets
)

func createSet(ctx context.Context, name search.SetName, dim, pre, num, batch int) (set search.Set, err error) {
	sets, err = NewSets(search.Config{
		Precision: 4,
		Dimension: 512,
		BlockSize: blockFeatureCount * dim * pre,
		BlockNum:  (num + blockFeatureCount - 1) / blockFeatureCount,
		BatchSize: batch,
	})
	if err != nil {
		return
	}

	err = sets.New(ctx, name, search.Config{Dimension: dim, Precision: pre, Version: 0}, SetStateCreated)
	if err != nil {
		return
	}

	var (
		features []proto.Feature
	)
	for i := 0; i < num; i++ {
		feature := proto.Feature{
			ID: proto.FeatureID(search.GetRandomString(12)),
		}
		r := rand.New(rand.NewSource(time.Now().Unix()))
		for j := 0; j < dim; j++ {
			bs := make([]byte, 4)
			binary.LittleEndian.PutUint32(bs, math.Float32bits(r.Float32()*2-1))
			feature.Value = append(feature.Value, proto.FeatureValue(bs)...)
		}
		features = append(features, feature)
	}

	set, err = sets.Get(ctx, name)
	if err != nil {
		return
	}
	err = set.Add(ctx, features...)
	if err != nil {
		return
	}
	return
}

func createFeatureValues(ctx context.Context, dim, num int) (targets []proto.FeatureValue) {
	for i := 0; i < num; i++ {
		feature := proto.Feature{
			ID: proto.FeatureID(search.GetRandomString(12)),
		}
		r := rand.New(rand.NewSource(time.Now().Unix()))
		for j := 0; j < dim; j++ {
			bs := make([]byte, 4)
			binary.LittleEndian.PutUint32(bs, math.Float32bits(r.Float32()*2-1))
			feature.Value = append(feature.Value, proto.FeatureValue(bs)...)
		}
		targets = append(targets, feature.Value)
	}
	return
}

func TestSets(t *testing.T) {
	ctx := context.Background()
	setName := search.SetName("test")

	set, err := createSet(ctx, setName, 5, 4, 10, 10)
	assert.Nil(t, err)

	results, err := set.Search(ctx, -1.0, 1, createFeatureValues(ctx, 5, 10)...)
	assert.Nil(t, err)
	for _, result := range results {
		assert.Equal(t, len(result), 1)
	}

	id := sets.AllBlocks[0].IDs[0]
	deleted, err := set.Delete(ctx, []proto.FeatureID{id}...)
	assert.Nil(t, err)
	assert.Equal(t, 1, len(deleted))
	assert.Equal(t, string(id), string(deleted[0]))
	assert.Nil(t, set.Destroy(ctx))

	_, err = set.Delete(ctx, []proto.FeatureID{proto.FeatureID(""), id}...)
	assert.Equal(t, search.ErrDeleteEmptyID, err)

	//check whether the set is really deleted by tring to get it after calling Destroy()
	_, err = sets.Get(ctx, setName)
	assert.Equal(t, err, search.ErrFeatureSetNotFound)
}

// benchmark cmd
// go test -v -bench='.'  -benchtime=10s qiniu.com/argus/feature_group_private/search/cpu
func BenchmarkBaseSearch1TO100000(b *testing.B) {
	ctx := context.Background()
	// memory benchmark
	b.ReportAllocs()
	b.SetBytes(2)

	set, err := createSet(ctx, search.SetName("test"), 512, 4, 10000, 10)
	assert.Nil(b, err)
	targets := createFeatureValues(ctx, 512, 1)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err = set.Search(ctx, -1.0, 1, targets...)
		assert.Nil(b, err)
	}
}
