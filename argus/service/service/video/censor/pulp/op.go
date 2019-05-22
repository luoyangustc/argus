package pulp

import (
	"context"

	"github.com/go-kit/kit/endpoint"

	"qiniu.com/argus/sdk/video"
	"qiniu.com/argus/service/service/image"
	pimage "qiniu.com/argus/service/service/image"
	"qiniu.com/argus/service/service/image/pulp"
	"qiniu.com/argus/service/service/video/censor"
	"qiniu.com/argus/utility/evals"
	video0 "qiniu.com/argus/video"
	"qiniu.com/argus/video/vframe"
)

type PulpService interface {
	Pulp(context.Context, pulp.PulpReq) (evals.PulpResp, error)
}

func NewOP(gen func() endpoint.Endpoint, config pimage.SugConfig) censor.OPFactory {
	return censor.SpecialCutOPFactory{
		CutParamFunc: func(_ context.Context, origin censor.CutParam) *censor.CutParam {
			if origin.Mode != vframe.MODE_INTERVAL {
				return nil
			}
			var intervalMs = video0.GCD(origin.IntervalMsecs, 500)
			return &censor.CutParam{
				Mode:          origin.Mode,
				IntervalMsecs: intervalMs,
			}
		},
		NewCutsFunc: func(
			ctx context.Context,
			cutParam censor.CutParam,
			options ...video.CutOpOption,
		) (video.CutsPipe, error) {

			eval_ := gen()
			eval := func(ctx context.Context, body []byte) (evals.PulpResp, error) {
				req := pulp.PulpReq{}
				req.Data.IMG.URI = image.DataURI(body)
				resp, err := eval_(ctx, req)
				if err != nil {
					return evals.PulpResp{}, err
				}
				return resp.(evals.PulpResp), nil
			}

			var func_ func(context.Context, *video.Cut) (interface{}, error)
			var options1 []video.CutOpOption
			if cutParam.Mode == vframe.MODE_INTERVAL {
				func_, options1 = NewModeIntervalOPFunc(eval)
			} else {
				func_, options1 = NewModeKeyOPFunc(eval)
			}
			options = append(options, options1...)

			func1 := func(ctx context.Context, cut *video.Cut) (interface{}, error) {
				resp_, err := func_(ctx, cut)
				if err != nil {
					return nil, err
				}

				resp := resp_.(evals.PulpResp)
				ret := pulp.ConvPulp(ctx, &config, resp)
				return ret, nil
			}

			return video.CreateCutOP(func1, options...)
		},
	}
}

type PulpOP struct {
	PulpService
}

func NewPulpOP(s PulpService) PulpOP {
	return PulpOP{PulpService: s}
}

func (op PulpOP) NewEval() endpoint.Endpoint {
	return func(ctx context.Context, req_ interface{}) (interface{}, error) {
		req := req_.(pulp.PulpReq)
		return op.eval(ctx, req)
	}
}

func (op PulpOP) eval(ctx context.Context, req pulp.PulpReq) (evals.PulpResp, error) {
	resp, err := op.PulpService.Pulp(ctx, req)
	if err != nil {
		return evals.PulpResp{}, err
	}
	return resp, nil
}

func NewModeIntervalOPFunc(eval func(context.Context, []byte) (evals.PulpResp, error)) (
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

func NewModeKeyOPFunc(eval func(context.Context, []byte) (evals.PulpResp, error)) (
	func(context.Context, *video.Cut) (interface{}, error),
	[]video.CutOpOption,
) {
	return func(ctx context.Context, cut *video.Cut) (interface{}, error) {
		body, _ := cut.Body()
		return eval(ctx, body)
	}, []video.CutOpOption{}
}

func opFunc(ctx context.Context, cut *video.Cut) (interface{}, error) {
	var scores = make(map[int]struct {
		Class  string
		Scores []float32
	})

	var rets = make([]evals.PulpResp, 0, 3)
	getResp := func(offset int64, tag string) error {
		ret, err := cut.GetRoundResp(offset, tag)
		if err != nil {
			return err
		}
		if ret != nil {
			rets = append(rets, ret.(evals.PulpResp))
		}
		return nil
	}
	if err := getResp(-500, ""); err != nil {
		return nil, err
	}
	if err := getResp(0, ""); err != nil {
		return nil, err
	}
	if err := getResp(500, ""); err != nil {
		return nil, err
	}

	for _, ret := range rets {
		for _, c := range ret.Result.Confidences {
			if ss, ok := scores[c.Index]; ok {
				ss.Scores = append(ss.Scores, c.Score)
				scores[c.Index] = ss
			} else {
				scores[c.Index] = struct {
					Class  string
					Scores []float32
				}{
					Class:  c.Class,
					Scores: []float32{c.Score},
				}
			}
		}
	}

	ret := evals.PulpResp{}
	for i, ss := range scores {
		var sum float32
		for _, s := range ss.Scores {
			sum += s
		}
		var avg = sum / float32(len(ss.Scores))

		ret.Result.Confidences = append(ret.Result.Confidences, struct {
			Index int     `json:"index"`
			Class string  `json:"class"`
			Score float32 `json:"score"`
		}{
			Index: i,
			Class: ss.Class,
			Score: avg,
		})
	}

	return ret, nil
}
