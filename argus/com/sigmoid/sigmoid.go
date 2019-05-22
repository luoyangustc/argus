package sigmoid

import (
	"math"
)

// g(x) = sigmoid(a(x-b+c))
// f(x) = g(x), if x<x0
//        x,    if x>=x0
//
// a: 决定了sigmoid函数的中间斜线斜率，取[5,10]，取值越大斜率越大
// b: 变化前的判别阈值thr_old,例如facex-feature-v3推荐为0.4
// c: sigmoid(ac)=变换后的判别阈值thr_new,ac=-ln(1/thr_new - 1);如果希望变换后阈值为0.5，则ac=0(a=6时c=0);
//    如果希望thr_new为0.65，则ac=0.619(a=6时,c=0.103)
// x0: x0是g(x)=x时x的取值。靠近x=1.0是，g(x)<x,此时取f(x)=x,可保证变换后区间取到1.0

const (
	DefaultSigmoidA = 10.0
)

type SigmoidConfig struct {
	ThresholdNew float64 `json:"threshold_new"`
	SigmoidA     float64 `json:"sigmoid_a"`
}

func Sigmoid(x, threshold, target_threshold, sigmoidA float64) float64 {
	var (
		a float64 = sigmoidA
		b float64 = threshold
		c float64
	)
	if a <= 0 {
		a = DefaultSigmoidA
	}
	if target_threshold == 0 {
		return 0
	}
	c = math.Log(1.0/target_threshold-1) * (-1.0) / a
	y := 1.0 / (1.0 + math.Exp((-1.0)*a*(x-b+c)))
	if y <= x {
		return x
	}
	return y
}
