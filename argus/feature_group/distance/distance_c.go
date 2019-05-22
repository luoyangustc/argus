// +build clangcgo

package distance

/*
#cgo CFLAGS: -ffast-math
#include <math.h>
float searchDistance(void *a, void *b, int n){
	float* aa = (float*)a;
	float* bb = (float*)b;
	float sum = 0;
	#pragma clang loop vectorize(enable) interleave(enable)
	for(int i=0;i<n;i++){
		float cc = aa[i]-bb[i];
		sum+=cc*cc;
	}
	return sum;
}

float searchDistanceCosine(void *a, void *b, int n){
	float* aa = (float*)a;
	float* bb = (float*)b;
	float sum = 0;
	#pragma clang loop vectorize(enable) interleave(enable)
	for(int i=0;i<n;i++){
		sum += aa[i]*bb[i];
	}
	if(isnan(sum)){
		return -1;
	}
	return sum;
}

void searchDistancesCosine(void *a, void *b, int l, void *c, int n){
	float* aa = (float*)a;
	float* bb = (float*)b;
	float* cc = (float*)c;
	float sum = 0;
	#pragma clang loop vectorize(enable) interleave(enable)
	for(int j=0;j<n;j++) {
		sum = 0;
		for(int i=0;i<l;i++){
			sum += aa[i]*bb[j*l+i];
		}
		if(isnan(sum)){
			sum = -1;
		}
		cc[j] = sum;
	}
}
*/
import "C"
import (
	"unsafe"
)

// calculate the Euclidean Distance
func DistanceCgo(feature1, feature2 []byte) float32 {
	r := C.searchDistance(unsafe.Pointer(&feature1[0]), unsafe.Pointer(&feature2[0]), C.int(len(feature1)/4))
	return float32(r)
}

// calculate the Euclidean Distance
func DistanceCosineCgo(feature1, feature2 []byte) float32 {
	r := C.searchDistanceCosine(unsafe.Pointer(&feature1[0]), unsafe.Pointer(&feature2[0]), C.int(len(feature1)/4))
	return float32(r)
}

func DistancesCosineCgoFlat(feature []byte, raw_features []byte, r []float32) {
	count := len(raw_features) / len(feature)
	C.searchDistancesCosine(unsafe.Pointer(&feature[0]), unsafe.Pointer(&raw_features[0]), C.int(len(feature)/4), unsafe.Pointer(&r[0]), C.int(count))
	return
}

func DistancesCosineCgo(feature []byte, features [][]byte) []float32 {
	raw_features := make([]byte, 0, len(features)*len(feature))
	for _, f := range features {
		raw_features = append(raw_features, f...)
	}
	r := make([]float32, len(features))
	C.searchDistancesCosine(unsafe.Pointer(&feature[0]), unsafe.Pointer(&raw_features[0]), C.int(len(feature)/4), unsafe.Pointer(&r[0]), C.int(len(features)))
	return r
}
