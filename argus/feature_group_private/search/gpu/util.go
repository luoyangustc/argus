// +build cublas

package gpu

import (
	"math/rand"
	"sort"
	"time"
	"unsafe"

	"github.com/pkg/errors"
	"qiniu.com/argus/feature_group_private"
	"qiniu.com/argus/feature_group_private/proto"
)

func Float32Array2DTo1D(array [][]float32) (ret []float32) {
	for _, row := range array {
		ret = append(ret, row...)
	}
	return
}

func FeautureTOFeatureValue(features []proto.Feature) (ret proto.FeatureValue) {
	for _, feature := range features {
		ret = append(ret, feature.Value...)
	}
	return
}

// 二维特征矩阵 转置成 一维列优先矩阵
func TFloat32(vector [][]float32, col, row int) []float32 {
	ret := make([]float32, col*row)
	for i := 0; i < col; i++ {
		for j := 0; j < row; j++ {
			ret[i*row+j] = vector[j][i]
		}
	}
	return ret
}

func MaxFloat32(vector []float32) (int, float32) {
	var max float32
	max = -99999.9
	var index int
	index = -1
	for i, value := range vector {
		if value > max {
			index = i
			max = value
		}
	}
	return index, max
}

func MaxNFloat32(vector []float32, limit int) ([]int, []float32) {
	type _result struct {
		Value float32
		Index int
	}
	var scores []_result
	for k, v := range vector {
		scores = append(scores, _result{Value: v, Index: k})
	}
	sort.Slice(scores, func(i, j int) bool { return scores[i].Value > scores[j].Value })
	var index []int
	var max []float32
	for i := 0; i < limit && i < len(scores); i++ {
		index = append(index, scores[i].Index)
		max = append(max, scores[i].Value)
	}
	return index, max
}

func MaxNFeatureResult(vector []feature_group.FeatureSearchItem, limit int) ([]int, []feature_group.FeatureSearchItem) {
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

// 字节序和 https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.tobytes.html 保持一致，即小端字节序
func ToFloat32Array(buf []byte) ([]float32, error) {
	l := len(buf)
	if l%4 != 0 {
		return nil, errors.New("ToFloat32Array bad length")
	}
	return (*(*[]float32)(unsafe.Pointer(&buf)))[:len(buf)/4], nil
}
