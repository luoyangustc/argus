// +build !clangcgo

package distance

import (
	"encoding/binary"
	"math"
)

// calculate the Euclidean Distance
func DistanceCosineCgo(feature1, feature2 []byte) (ret float32) {
	if len(feature1) != len(feature2) {
		panic("feature1 mismatch feature2")
	}
	for i := 0; i < len(feature1); i = i + 4 {
		a := math.Float32frombits(binary.LittleEndian.Uint32(feature1[i:(i + 4)]))
		b := math.Float32frombits(binary.LittleEndian.Uint32(feature2[i:(i + 4)]))
		ret += a * b
	}
	return
}

func DistancesCosineCgo(feature []byte, features [][]byte) []float32 {
	r := make([]float32, len(features))
	for id, f := range features {
		for i := 0; i < len(feature); i = i + 4 {
			a := math.Float32frombits(binary.LittleEndian.Uint32(feature[i:(i + 4)]))
			b := math.Float32frombits(binary.LittleEndian.Uint32(f[i:(i + 4)]))
			r[id] += a * b
		}
	}
	return r
}

func DistancesCosineCgoFlat(feature []byte, raw_features []byte, r []float32) {
	length := len(feature)
	count := len(raw_features) / length
	for id := 0; id < count; id++ {
		offset := length * id
		var score float32
		for i := 0; i < len(feature); i = i + 4 {
			a := math.Float32frombits(binary.LittleEndian.Uint32(feature[i:(i + 4)]))
			b := math.Float32frombits(binary.LittleEndian.Uint32(raw_features[(offset + i):(offset + i + 4)]))
			score += a * b
		}
		r[id] = score
	}
}
