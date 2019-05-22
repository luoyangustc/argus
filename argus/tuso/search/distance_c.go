// +build clangcgo

package search

/*
#include <math.h>
float distance(void *a, void *b, int n){
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

float distanceCosine(void *a, void *b, int n){
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
*/
import "C"
import (
	"unsafe"

	"qiniu.com/argus/tuso/proto"
)

// calculate the Euclidean Distance
func distanceCgo(feature1, feature2 proto.Feature) float32 {
	r := C.distance(unsafe.Pointer(&feature1[0]), unsafe.Pointer(&feature2[0]), C.int(len(feature1)/4))
	return float32(r)
}

// calculate the Euclidean Distance
func distanceCosineCgo(feature1, feature2 proto.Feature) float32 {
	r := C.distanceCosine(unsafe.Pointer(&feature1[0]), unsafe.Pointer(&feature2[0]), C.int(len(feature1)/4))
	return float32(r)
}

func init() {
	searchFuncs = append(searchFuncs, searchFunc{distanceCgo, "distanceCgo", 3, false})
	searchFuncs = append(searchFuncs, searchFunc{distanceCosineCgo, "distanceCosineCgo", 13, true})
}
