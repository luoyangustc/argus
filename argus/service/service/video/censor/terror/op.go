package terror

import (
	"context"

	"github.com/go-kit/kit/endpoint"

	"qiniu.com/argus/sdk/video"
	"qiniu.com/argus/service/service/image"
	iterror "qiniu.com/argus/service/service/image/terror"
	"qiniu.com/argus/service/service/video/censor"
	"qiniu.com/argus/utility/evals"
	video0 "qiniu.com/argus/video"
	"qiniu.com/argus/video/vframe"
)

type TerrorService interface {
	TerrorCensor(ctx context.Context, req iterror.SimpleReq) (res TerrorResp, err error)
}

type TerrorResp struct {
	MixResp    *iterror.TerrorMixupResp
	DetectResp *evals.TerrorDetectResp
}

func NewOP(genMain, genAuxiliary func() endpoint.Endpoint, config image.SugConfig) censor.OPFactory {
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
			evalMain_ := genMain()
			evalAuxiliary_ := genAuxiliary()

			evalMain := func(ctx context.Context, body []byte) (TerrorResp, error) {
				req := iterror.SimpleReq{}
				req.Data.IMG.URI = image.DataURI(body)
				resp, err := evalMain_(ctx, req)
				if err != nil {
					return TerrorResp{}, err
				}
				return resp.(TerrorResp), nil
			}
			evalAuxiliary := func(ctx context.Context, body []byte) (TerrorResp, error) {
				req := iterror.SimpleReq{}
				req.Data.IMG.URI = image.DataURI(body)
				resp, err := evalAuxiliary_(ctx, req)
				if err != nil {
					return TerrorResp{}, err
				}
				return resp.(TerrorResp), nil
			}

			var func_ func(context.Context, *video.Cut) (interface{}, error)
			var options1 []video.CutOpOption
			if cutParam.Mode == vframe.MODE_INTERVAL {
				func_, options1 = NewModeIntervalOPFunc(evalMain, evalAuxiliary)
			} else {
				func_, options1 = NewModeKeyOPFunc(evalMain)
			}
			options = append(options, options1...)

			func1 := func(ctx context.Context, cut *video.Cut) (interface{}, error) {
				resp_, err := func_(ctx, cut)
				if err != nil {
					return nil, err
				}

				resp := resp_.(TerrorResp)
				if len(resp.MixResp.Result.Confidences) == 0 {
					return image.SceneResult{}, nil
				}
				sr1 := iterror.ConvTerrorMixup(ctx, &config, *resp.MixResp)
				if resp.DetectResp == nil {
					return *sr1, nil
				}
				sr2 := iterror.ConvTerrorDetect(ctx, &config, *resp.DetectResp)
				sr3 := image.MergeDetails(ctx, &config, sr1.Details, sr2.Details...)
				return *sr3, nil
			}
			return video.CreateCutOP(func1, options...)
		},
	}
}

func NewModeIntervalOPFunc(evalMain, evalAuxiliary func(context.Context, []byte) (TerrorResp, error)) (
	func(context.Context, *video.Cut) (interface{}, error),
	[]video.CutOpOption,
) {
	f1 := func(ctx context.Context, req video.CutRequest) (interface{}, error) {
		return evalMain(ctx, req.Body)
	}
	f2 := func(ctx context.Context, req video.CutRequest) (interface{}, error) {
		return evalAuxiliary(ctx, req.Body)
	}
	return opFunc, []video.CutOpOption{
		video.WithRoundCutOP(-500, "auxiliary", f2),
		video.WithRoundCutOP(0, "main", f1),
		video.WithRoundCutOP(500, "auxiliary", f2),
	}
}

func NewModeKeyOPFunc(evalMain func(context.Context, []byte) (TerrorResp, error)) (
	func(context.Context, *video.Cut) (interface{}, error),
	[]video.CutOpOption,
) {
	return func(ctx context.Context, cut *video.Cut) (interface{}, error) {
		body, _ := cut.Body()
		return evalMain(ctx, body)
	}, []video.CutOpOption{}
}

func opFunc(ctx context.Context, cut *video.Cut) (interface{}, error) {
	var scores = make(map[string]struct {
		Scores []float32
		Index  int
	})

	var rets = make([]TerrorResp, 0, 3)
	getResp := func(offset int64, tag string) error {
		ret, err := cut.GetRoundResp(offset, tag)
		if err != nil {
			return err
		}
		if ret != nil {
			rets = append(rets, ret.(TerrorResp))
		}
		return nil
	}
	if err := getResp(-500, "auxiliary"); err != nil {
		return nil, err
	}
	if err := getResp(0, "main"); err != nil {
		return nil, err
	}
	if err := getResp(500, "auxiliary"); err != nil {
		return nil, err
	}

	resp := TerrorResp{MixResp: &iterror.TerrorMixupResp{}}
	for _, ret := range rets {
		if ret.DetectResp != nil {
			resp.DetectResp = ret.DetectResp
		}
		if ret.MixResp == nil {
			continue
		}
		for _, c := range ret.MixResp.Result.Confidences {
			if ss, ok := scores[c.Class]; ok {
				ss.Scores = append(ss.Scores, c.Score)
				scores[c.Class] = ss
			} else {
				scores[c.Class] = struct {
					Scores []float32
					Index  int
				}{
					Index:  c.Index,
					Scores: []float32{c.Score},
				}
			}
		}
	}
	for class, ss := range scores {
		var sum float32
		for _, s := range ss.Scores {
			sum += s
		}
		var avg = sum / float32(len(ss.Scores))
		resp.MixResp.Result.Confidences = append(resp.MixResp.Result.Confidences, struct {
			Index int     `json:"index"`
			Class string  `json:"class"`
			Score float32 `json:"score"`
		}{
			Index: ss.Index,
			Class: class,
			Score: avg,
		})
	}
	return resp, nil
}

type TerrorOP struct {
	TerrorService
}

func NewTerrorOP(s TerrorService) TerrorOP { return TerrorOP{TerrorService: s} }

func (op TerrorOP) NewEval() endpoint.Endpoint {
	return func(ctx context.Context, req0 interface{}) (interface{}, error) {
		req := req0.(iterror.SimpleReq)
		return op.eval(ctx, req)
	}
}

func (op TerrorOP) eval(ctx context.Context, req iterror.SimpleReq) (TerrorResp, error) {
	resp, err := op.TerrorService.TerrorCensor(ctx, req)
	if err != nil {
		return TerrorResp{}, err
	}
	return resp, nil
}
