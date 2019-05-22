package feature_group

import (
	"context"
	"encoding/binary"
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
)

type stringKey string

func (key stringKey) Key() string { return string(key) }

func TestWorker(t *testing.T) {

	formatFloat32 := func(fs []float32) []byte {
		bs := make([]byte, len(fs)*4)
		for i, f := range fs {
			binary.LittleEndian.PutUint32(bs[i*4:], math.Float32bits(f))
		}
		return bs
	}

	var bs []byte

	config := StorageConfig{Size: 1024, ChunkSize: 64}
	config.Slabs = append(config.Slabs,
		struct {
			Size  uint64 `json:"size"`
			Count int    `json:"count"`
		}{Size: 16, Count: 8},
		struct {
			Size  uint64 `json:"size"`
			Count int    `json:"count"`
		}{Size: 64, Count: 2})
	worker := Worker{
		NewMemory(config,
			func(context.Context, Key) ([]byte, error) { return bs, nil },
		)}

	bs = formatFloat32([]float32{0, 0, 1.0, 0, 0.5, 0})
	results, _ := worker.Search(
		context.Background(), stringKey("a"),
		formatFloat32([]float32{1.0, 0}), 2*4,
		0, 2,
	)

	assert.Equal(t, 1, len(results))
	assert.Equal(t, 2, len(results[0].Items))
	assert.Equal(t, 1, results[0].Items[0].Index)
	assert.Equal(t, 2, results[0].Items[1].Index)

	bs = formatFloat32([]float32{0, 0, 0.5, 0, 1.0, 0})
	results, _ = worker.Search(
		context.Background(), stringKey("a"),
		formatFloat32([]float32{1.0, 0}), 2*4,
		0, 2,
	)
	assert.Equal(t, 1, len(results))
	assert.Equal(t, 2, len(results[0].Items))
	assert.Equal(t, 1, results[0].Items[0].Index)
	assert.Equal(t, 2, results[0].Items[1].Index)

	bs = formatFloat32([]float32{
		0, 0, 0,
		1, 0, 0,
		0, 0, 0,
		0, 0, 0,
	})
	results, _ = worker.Search(
		context.Background(), stringKey("b"),
		formatFloat32([]float32{1.0, 0, 0}), 3*4,
		0, 1,
	)
	assert.Equal(t, 1, len(results))
	assert.Equal(t, 1, len(results[0].Items))
	assert.Equal(t, 1, results[0].Items[0].Index)

	results, _ = worker.Search(
		context.Background(), stringKey("a"),
		formatFloat32([]float32{1.0, 0}), 2*4,
		0, 2,
	)
	assert.Equal(t, 1, len(results))
	assert.Equal(t, 2, len(results[0].Items))
	assert.Equal(t, 1, results[0].Items[0].Index)
	assert.Equal(t, 2, results[0].Items[1].Index)

	bs = formatFloat32([]float32{
		0, 0, 0,
		0, 0, 0,
		1, 0, 0,
		0.5, 0, 0,
	})
	results, _ = worker.Search(
		context.Background(), stringKey("c"),
		formatFloat32([]float32{1.0, 0, 0}), 3*4,
		0, 2,
	)
	assert.Equal(t, 1, len(results))
	assert.Equal(t, 2, len(results[0].Items))
	assert.Equal(t, 2, results[0].Items[0].Index)
	assert.Equal(t, 3, results[0].Items[1].Index)

	bs = formatFloat32([]float32{0, 0, 0.5, 0, 1.0, 0})
	results, _ = worker.Search(
		context.Background(), stringKey("a"),
		formatFloat32([]float32{1.0, 0}), 2*4,
		0, 2,
	)
	assert.Equal(t, 1, len(results))
	assert.Equal(t, 2, len(results[0].Items))
	assert.Equal(t, 2, results[0].Items[0].Index)
	assert.Equal(t, 1, results[0].Items[1].Index)

}
