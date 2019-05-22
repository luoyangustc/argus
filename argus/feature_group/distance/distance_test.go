package distance

import (
	"encoding/binary"
	"math"
	"math/rand"
	"testing"
	"time"
	"unsafe"

	"github.com/stretchr/testify/assert"
	"qiniu.com/argus/feature_group_private/proto"
)

var (
	float32Width = 4
	size         = 512
	total        = 30000
	values1      []byte
	values2      [][]byte
	values3      []byte
)

func init() {
	r := rand.New(rand.NewSource(time.Now().Unix()))

	for j := 0; j < size; j++ {
		bs := make([]byte, float32Width)
		binary.LittleEndian.PutUint32(bs, math.Float32bits(r.Float32()*2-1))
		values1 = append(values1, bs...)
	}
	norm(values1)

	values2 = make([][]byte, total)
	values3 = make(proto.FeatureValue, 0)
	for i := 0; i < total; i++ {
		value := make([]byte, 0)
		for j := 0; j < size; j++ {
			bs := make([]byte, float32Width)
			binary.LittleEndian.PutUint32(bs, math.Float32bits(r.Float32()*2-1))
			value = append(value, bs...)
		}
		norm(value)
		values2[i] = append(values2[i], value...)
		values3 = append(values3, value...)
	}
}

func norm(a []byte) {
	r := (*(*[]float32)(unsafe.Pointer(&a)))[:len(a)/4]
	var sum float32
	for i := 0; i < len(r); i++ {
		sum += r[i] * r[i]
	}
	sum = float32(math.Sqrt(float64(sum)))
	for i := 0; i < len(r); i++ {
		r[i] /= sum
	}
}

func equal(f1, f2 []float32) bool {
	if len(f1) != len(f2) {
		return false
	}
	for id := range f1 {
		if f1[id] != f2[id] {
			return false
		}
	}
	return true
}

func TestResult(t *testing.T) {
	DistancesCosineCgo(values1, values2)
	r2 := make([]float32, len(values3)/len(values1))
	DistancesCosineCgoFlat(values1, values3, r2)
	a := assert.New(t)
	r1 := DistancesCosineCgo(values1, values2)
	DistancesCosineCgoFlat(values1, values3, r2)
	a.True(equal(r1, r2))
}

func BenchmarkGO(b *testing.B) {
	for i := 0; i < b.N; i++ {
		DistancesCosineCgo(values1, values2)
	}
}

func BenchmarkGOFlat(b *testing.B) {
	scores := make([]float32, len(values3)/len(values1))
	for i := 0; i < b.N; i++ {
		DistancesCosineCgoFlat(values1, values3, scores)
	}
}
