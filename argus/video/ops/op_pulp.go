package ops

import (
	"context"
	"os"
	"strconv"
	"time"

	"github.com/qiniu/xlog.v1"

	ahttp "qiniu.com/argus/argus/com/http"
	"qiniu.com/argus/video"
	"qiniu.com/argus/video/vframe"
)

func RegisterPulp() {
	video.RegisterOP("pulp",
		func(config video.OPConfig) video.OP { return iPulp{OPConfig: config} })
	video.RegisterOP("pulp_beta",
		func(config video.OPConfig) video.OP { return iPulp{OPConfig: config, useBeta: true} })
}

var _ video.OP = iPulp{}
var _ video.CutOP = iPulp{}
var _ video.SpecialCutOP = iPulp{}

type iPulp struct {
	video.OPConfig
	video.OPEnv
	timeout time.Duration
	useBeta bool
}

func (p iPulp) Fork(ctx context.Context, params video.OPParams, env video.OPEnv) (video.OP, bool) {
	return iPulp{
		OPConfig: video.OPConfig{
			Host:   p.OPConfig.Host,
			Params: params,
		},
		OPEnv: env,
	}, true
}

func (p iPulp) Reset(context.Context) error { return nil }
func (p iPulp) Count() int32                { return video.MAX_OP_COUNT }

func (p iPulp) VframeParams(
	ctx context.Context, origin vframe.VframeParams,
) *vframe.VframeParams {

	if origin.GetMode() != vframe.MODE_INTERVAL {
		return nil
	}
	var interval = video.GCD(int64(1000*origin.Interval), 500)
	return &vframe.VframeParams{
		Mode:     origin.Mode,
		Interval: float64(interval) / 1000,
	}
}

func (p iPulp) NewCuts(
	ctx context.Context, params, OriginParams *vframe.VframeParams,
) video.CutsV2 {

	type EvalResult struct {
		Label       int     `json:"label"`
		Score       float32 `json:"score"`
		Review      bool    `json:"review"`
		Confidences []struct {
			Index int     `json:"index"`
			Class string  `json:"class"`
			Score float32 `json:"score"`
		} `json:"confidences,omitempty"`
	}

	type EvalResp struct {
		Code    int        `json:"code"`
		Message string     `json:"message"`
		Result  EvalResult `json:"result"`
	}

	type EvalReq struct {
		Data struct {
			URI string `json:"uri"`
		} `json:"data"`
		Params struct {
			Limit int `json:"limit"`
		} `json:"params"`
	}

	var (
		evalFunc = func(ctx context.Context, uris []string, free bool) ([]interface{}, error) {
			if os.Getenv("USE_SERVING") == "TRUE" {
				return p.servingEval(ctx, uris)
			}
			utype := p.OPEnv.Utype
			if free {
				utype = 0
			}

			var (
				xl     = xlog.FromContextSafe(ctx)
				client = ahttp.NewQiniuStubRPCClient(p.OPEnv.Uid, utype, p.timeout)

				resps  = make([]EvalResp, 0)
				result = make([]interface{}, 0)
				url    = p.OPConfig.Host + "/v1/pulp"
			)
			if p.useBeta {
				url = p.OPConfig.Host + "/v1/beta/pulp"
			}

			xl.Infof("query pulp: %d", len(uris))

			for _, uri := range uris { // TODO 并行
				xl.Infof("url and data: %#v, %#v", url, video.PrintableURI(uri))
				var resp EvalResp
				f := func(ctx context.Context) error {
					req := EvalReq{}
					req.Data.URI = uri
					req.Params.Limit = 3
					return client.CallWithJson(ctx, &resp, "POST", url, req)
				}
				if err := ahttp.CallWithRetry(
					ctx, []int{530}, []func(context.Context) error{f, f},
				); err != nil {
					xl.Errorf("query pulp failed. error: %v, resp: %#v", err, resp)
					return nil, err //失败则返回失败
				}
				resps = append(resps, resp)
			}

			xl.Infof("query pulp done. %#v", resps)

			for _, resp := range resps {
				if resp.Code != 0 {
					xl.Warnf("pulp cut failed. %#v", resp)
					result = append(result, nil)
					continue
				}
				result = append(result, resp.Result)
			}

			return result, nil
		}

		parseFunc = func(ctx context.Context, _v interface{}) (
			labels []string, scores []float32, err error,
		) {
			if _v == nil {
				return nil, nil, nil
			}

			if v, ok := _v.(EvalResult); ok {
				labels = []string{strconv.Itoa(v.Label)}
				scores = []float32{v.Score}
			} else {
				v := _v.(struct {
					Confidences []struct {
						Index int     `json:"index"`
						Class string  `json:"class"`
						Score float32 `json:"score"`
					} `json:"confidences"`
				})
				for i := range v.Confidences {
					labels = append(labels, v.Confidences[i].Class)
					scores = append(scores, v.Confidences[i].Score)
				}
			}
			return
		}
	)
	if params == nil || params.GetMode() != vframe.MODE_INTERVAL {
		return video.NewSimpleCutsV2(nil, nil,
			func(ctx context.Context, uris []string) ([]interface{}, error) {
				return evalFunc(ctx, uris, false)
			},
			parseFunc)
	}
	cuts := video.NewFixedAggCutsV2(
		int64(params.Interval*1000),
		int64(OriginParams.Interval*1000),
		[]int64{-500, 0, 500},
		evalFunc,
		func(ctx context.Context, results []interface{}) (interface{}, error) {

			var scores = make(map[int][]float32)
			var _confidences []struct {
				Index int     `json:"index"`
				Class string  `json:"class"`
				Score float32 `json:"score"`
			}
			for _, r := range results {
				if r == nil {
					continue
				}

				if os.Getenv("USE_SERVING") == "TRUE" {
					_rest := r.(struct {
						Confidences []struct {
							Index int     `json:"index"`
							Class string  `json:"class"`
							Score float32 `json:"score"`
						} `json:"confidences"`
					})
					_confidences = _rest.Confidences
				} else {
					_rest := r.(EvalResult)
					_confidences = _rest.Confidences
				}
				for _, c := range _confidences {
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

			evalResult := EvalResult{Label: index, Score: max}

			// 合并平均帧之后，需要再重置review
			// Tips: 需要与argus-util中的值要保持一致
			if evalResult.Score < 0.6 {
				evalResult.Review = true
			}

			return evalResult, nil
		},
		parseFunc,
	)
	return cuts
}

func (p iPulp) servingEval(ctx context.Context, uris []string) ([]interface{}, error) {
	type EvalPulpResult struct {
		Code    int    `json:"code"`
		Message string `json:"message"`
		Result  struct {
			Confidences []struct {
				Index int     `json:"index"`
				Class string  `json:"class"`
				Score float32 `json:"score"`
			} `json:"confidences"`
		} `json:"result"`
	}

	var (
		xl     = xlog.FromContextSafe(ctx)
		client = ahttp.NewQiniuStubRPCClient(p.OPEnv.Uid, p.OPEnv.Utype, p.timeout)

		resps  = make([]EvalPulpResult, 0)
		result = make([]interface{}, 0)
		url    = p.OPConfig.Host + "/v1/eval/pulp"
	)

	xl.Infof("query pulp: %d", len(uris))

	for _, uri := range uris { // TODO 并行
		xl.Infof("url and data: %#v, %#v", url, video.PrintableURI(uri))
		var resp EvalPulpResult
		if err := client.CallWithJson(ctx, &resp, "POST",
			url,
			struct {
				Data struct {
					URI string `json:"uri"`
				} `json:"data"`
			}{
				Data: struct {
					URI string `json:"uri"`
				}{
					URI: uri,
				},
			},
		); err != nil {
			xl.Errorf("query pulp failed. error: %v, resp: %#v", err, resp)
			return nil, err //失败则返回失败
		}
		resps = append(resps, resp)
	}

	xl.Infof("query pulp done. %#v", resps)

	for _, resp := range resps {
		if resp.Code != 0 {
			xl.Warnf("pulp cut failed. %#v", resp)
			result = append(result, nil)
			continue
		}
		result = append(result, resp.Result)
	}

	return result, nil
}
