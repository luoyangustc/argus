package pulp

import (
	"context"

	"qiniu.com/argus/sdk/video"
	"qiniu.com/argus/service/service/image/pulp"
)

func NewOP(eval func(context.Context, []byte) (pulp.PulpResult, error)) (
	func(context.Context, *video.Cut) (interface{}, error),
	[]video.CutOpOption,
) {
	f := func(ctx context.Context, req video.CutRequest) (interface{}, error) {
		return eval(ctx, req.Body)
	}
	return opFunc, []video.CutOpOption{
		video.WithRoundCutOP(-500, "", f),
		video.WithRoundCutOP(0, "", f),
		video.WithRoundCutOP(500, "", f),
	}
}

func NewOPModeKey(eval func(context.Context, []byte) (pulp.PulpResult, error)) (
	func(context.Context, *video.Cut) (interface{}, error),
	[]video.CutOpOption,
) {
	return func(ctx context.Context, cut *video.Cut) (interface{}, error) {
		body, _ := cut.Body()
		return eval(ctx, body)
	}, []video.CutOpOption{}
}

func opFunc(ctx context.Context, cut *video.Cut) (interface{}, error) {
	var scores = make(map[int][]float32)

	var rets = make([]pulp.PulpResult, 0, 3)
	if ret, _ := cut.GetRoundResp(-500, ""); ret != nil {
		rets = append(rets, ret.(pulp.PulpResult))
	}
	if ret, _ := cut.GetRoundResp(0, ""); ret != nil {
		rets = append(rets, ret.(pulp.PulpResult))
	}
	if ret, _ := cut.GetRoundResp(500, ""); ret != nil {
		rets = append(rets, ret.(pulp.PulpResult))
	}
	for _, ret := range rets {
		for _, c := range ret.Confidences {
			if ss, ok := scores[c.Index]; ok {
				scores[c.Index] = append(ss, c.Score)
			} else {
				scores[c.Index] = []float32{c.Score}
			}
		}
	}
	var (
		index = -1
		max   float32
	)
	for i, ss := range scores {
		var sum float32
		for _, s := range ss {
			sum += s
		}
		var avg = sum / float32(len(ss))
		if avg > max {
			index = i
			max = avg
		}
	}

	evalResult := pulp.PulpResult{Label: index, Score: max}

	// 合并平均帧之后，需要再重置review
	// Tips: 需要与argus-util中的值要保持一致
	if evalResult.Score < 0.6 {
		evalResult.Review = true
	}

	return evalResult, nil
}
