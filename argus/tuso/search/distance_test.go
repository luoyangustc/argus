package search

import (
	"math"
	"math/rand"
	"testing"

	"github.com/stretchr/testify/assert"

	"qiniu.com/argus/tuso/proto"
)

func TestDistance(t *testing.T) {
	for _, searchFunc := range searchFuncs {
		t.Run(searchFunc.name, func(t *testing.T) {
			var tests = []struct {
				feature1       []float32
				feature2       []float32
				expected       float32
				expectedCosine float32
			}{
				{[]float32{1}, []float32{1}, 0, 1.0},
				{[]float32{1, 2, 3}, []float32{1, 2, 3}, 0, 1.0},
				{[]float32{1, 2, 3, 4, 5}, []float32{1, 2, 3, 4, 5}, 0, 1.0},
				{[]float32{2}, []float32{5}, 9, 1.0},
				{[]float32{1, 3, 5}, []float32{2, 4, 6}, 3, 0.993858693196},
				{[]float32{1, 2, 3, 4, 5}, []float32{5, 5, 5, 5, 5}, 30, 0.904534033733},
				{[]float32{1.5, 2.5, 3.5}, []float32{1.0, 2.1, 4.7}, 1.85, 0.971208752178},
				{[]float32{1, 3, 5}, []float32{0, 0, 0}, 35, -1},
			}

			for _, test := range tests {
				if searchFunc.isCosine {
					a := make([]float32, len(test.feature1))
					b := make([]float32, len(test.feature2))
					copy(a, test.feature1)
					copy(b, test.feature2)
					norm(a)
					norm(b)
					if dis := searchFunc.f(float32tobytes(a...), float32tobytes(b...)); math.Abs(float64(dis-test.expectedCosine)) > 0.000001 || math.IsNaN(float64(dis)) {
						t.Errorf("distance(%v, %v) = %f, expected: %f", test.feature1, test.feature2, dis, test.expectedCosine)
					}
				} else {
					if dis := searchFunc.f(float32tobytes(test.feature1...), float32tobytes(test.feature2...)); math.Abs(float64(dis-test.expected)) > 0.000001 || math.IsNaN(float64(dis)) {
						t.Errorf("distance(%v, %v) = %f, expected: %f", test.feature1, test.feature2, dis, test.expected)
					}
				}
			}

		})
	}
}

func BenchmarkDistance(b *testing.B) {
	for _, searchFunc := range searchFuncs {
		b.Run(searchFunc.name, func(b *testing.B) {
			count := FeaturesPerSet / 64
			features := prepareFeatureItems(count)

			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				x := rand.Int() % count
				y := rand.Int() % count
				_ = searchFunc.f(features[x].Feature, features[y].Feature)
			}
		})
	}
}

func TestNormFeatures(t *testing.T) {
	assert := assert.New(t)
	a := make([]byte, proto.FeatureSize*3)
	b := toFloat32Array(a)
	for i := 0; i < len(b); i++ {
		b[i] = float32(i)
	}
	NormFeatures(a, proto.FeatureSize)
	for i := 0; i < len(b); i++ {
		assert.True(b[i] >= 0)
		assert.True(b[i] <= 1)
	}
	assert.InDelta(b[2]/2, b[1], 0.001)
	var sum float32
	for i := 0; i < proto.FeatureSize/4; i++ {
		sum += b[i] * b[i]
	}
	assert.InDelta(sum, 1, 0.001)
}
