package feature_group

import (
	"context"
	"encoding/binary"
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestMemory(t *testing.T) {

	EF := func(a, b float32) bool {
		return math.Abs(float64(a-b)) < 0.000001
	}

	{
		mem := Memory{}

		items := mem.result(
			[]SearchResultItem{
				{Index: 1, Score: 0.5},
			},
			0, 2,
		)
		assert.Equal(t, 1, len(items))
		assert.Equal(t, 1, items[0].Index)
		assert.True(t, EF(0.5, items[0].Score))

		items = mem.result(
			[]SearchResultItem{
				{Index: 1, Score: 0.4},
				{Index: 3, Score: 0.2},
				{Index: 2, Score: 0.3},
			},
			0, 2,
		)
		assert.Equal(t, 2, len(items))
		assert.Equal(t, 1, items[0].Index)
		assert.True(t, EF(0.4, items[0].Score))
		assert.Equal(t, 2, items[1].Index)
		assert.True(t, EF(0.3, items[1].Score))
	}
}

func BenchmarkMemorySearch(b *testing.B) {

	mem := NewMemory(
		StorageConfig{
			Size:      1024 * 1024 * 64,
			ChunkSize: 1024 * 1024 * 64,
			Slabs: []struct {
				Size  uint64 `json:"size"`
				Count int    `json:"count"`
			}{{Size: 1024 * 1024 * 64, Count: 1}},
		},
		func(context.Context, Key) ([]byte, error) {
			var n = 64 * 1024 / 16
			var bs = make([]byte, 1024*16*n)
			// FormatFloat32s(binary.LittleEndian, make([]float32, 1024*4*n), bs)
			return bs, nil
		},
	)

	var bs = make([]byte, 1024*16)
	FormatFloat32s(binary.LittleEndian, make([]float32, 1024*4), bs)

	for i := 0; i < b.N; i++ {
		_, _ = mem.Search(context.Background(), SearchKey{}, bs, 1024*16, 0, 100)
	}
}
