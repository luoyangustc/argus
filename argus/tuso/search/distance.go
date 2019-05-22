package search

import (
	"fmt"
	"math"
	"unsafe"

	"qiniu.com/argus/tuso/proto"
	"qiniu.com/argus/tuso/utils"
)

type searchFunc struct {
	f        func(feature1, feature2 proto.Feature) float32
	name     string
	score    int
	isCosine bool
}

var searchFuncs []searchFunc

func init() {
	searchFuncs = append(searchFuncs, searchFunc{distanceOrigin, "distanceOrigin", 1, false})
	searchFuncs = append(searchFuncs, searchFunc{distanceUnsafe, "distanceUnsafe", 2, false})
	searchFuncs = append(searchFuncs, searchFunc{distanceCosine, "distanceCosine", 10, true})
}

// calculate the Euclidean Distance
func distanceOrigin(feature1, feature2 proto.Feature) float32 {
	// feature is float32
	count := len(feature1) / 4

	var ret float32
	for index := 0; index < count; index++ {
		start := index * 4
		feature1Float := utils.ByteToFloat32(feature1[start : start+4])
		feature2Float := utils.ByteToFloat32(feature2[start : start+4])
		ret += (feature1Float - feature2Float) * (feature1Float - feature2Float)
	}

	return ret
}

func toFloat32Array(buf []byte) []float32 {
	l := len(buf)
	if l%4 != 0 {
		panic("bad array")
	}
	return (*(*[]float32)(unsafe.Pointer(&buf)))[:len(buf)/4]
}

// calculate the Euclidean Distance
func distanceUnsafe(feature1, feature2 proto.Feature) float32 {
	// feature is float32
	count := len(feature1) / 4

	f1 := toFloat32Array(feature1)
	f2 := toFloat32Array(feature2)

	var ret float32
	for index := 0; index < count; index++ {
		r := (f1[index] - f2[index])
		ret += r * r
	}

	return ret
}

func BestCosineDistance() searchFunc {
	r := searchFunc{score: -1}
	for _, f := range searchFuncs {
		if f.isCosine && f.score > r.score {
			r = f
		}
	}
	return r
}

func distanceCosine(feature1, feature2 proto.Feature) float32 {
	count := len(feature1) / 4

	f1 := toFloat32Array(feature1)
	f2 := toFloat32Array(feature2)

	var sum float32
	for i := 0; i < count; i++ {
		sum += f1[i] * f2[i]
	}
	if math.IsNaN(float64(sum)) {
		return -1
	}
	return sum
}

func norm(a []float32) {
	var sum float32
	for i := 0; i < len(a); i++ {
		sum += a[i] * a[i]
	}
	sum = float32(math.Sqrt(float64(sum)))
	for i := 0; i < len(a); i++ {
		a[i] /= sum
	}
}

func NormFeatures(a []byte, featureByteSize int) {
	if (len(a))%featureByteSize != 0 || featureByteSize%4 != 0 {
		panic(fmt.Sprintf("bad feature %v %v", len(a), featureByteSize))
	}
	featureFloatSize := featureByteSize / 4
	r := toFloat32Array(a)
	for i := 0; i+featureFloatSize <= len(r); i += featureFloatSize {
		norm(r[i : i+featureFloatSize])
	}
}
