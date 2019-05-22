package ops

import (
	"context"
	"time"

	"github.com/qiniu/rpc.v3"
	"github.com/qiniu/xlog.v1"

	ahttp "qiniu.com/argus/argus/com/http"
	"qiniu.com/argus/video"
	"qiniu.com/argus/video/vframe"
)

type (
	CutEval func(
		ctx context.Context,
		cli *rpc.Client, host string,
		uri string,
	) (interface{}, error)

	NewCutEval func(video.OPParams) CutEval

	CutResults interface {
		Len() int
		Parse(int) (string, float32, bool)
	}

	ParseCutResults    func(interface{}) (CutResults, bool)
	NewParseCutResults func(video.OPParams) ParseCutResults
)

type SimpleCutOP struct {
	video.OPConfig
	video.OPEnv
	Pool    chan *SimpleCutOP
	timeout time.Duration

	newEval NewCutEval
	eval    CutEval

	newParse NewParseCutResults
	parse    ParseCutResults
}

func NewSimpleCutOP(config video.OPConfig, eval CutEval) SimpleCutOP {
	sc := SimpleCutOP{OPConfig: config, eval: eval}
	sc.initPool(config)
	return sc
}

func NewSimpleCutOP2(config video.OPConfig, newEval NewCutEval, newParse NewParseCutResults) SimpleCutOP {
	sc := SimpleCutOP{OPConfig: config, newEval: newEval, newParse: newParse}
	sc.initPool(config)
	return sc
}

func (sc *SimpleCutOP) initPool(config video.OPConfig) {
	if len(config.Instances) > 0 {
		sc.Pool = make(chan *SimpleCutOP, len(config.Instances))
		for _, host := range config.Instances {
			sc.Pool <- &SimpleCutOP{
				OPConfig: video.OPConfig{
					Host: host,
				},
				Pool: sc.Pool,
			}
		}
	}
}

func (sc SimpleCutOP) Fork(ctx context.Context, params video.OPParams, env video.OPEnv) (video.OP, bool) {
	var op *SimpleCutOP
	if sc.Pool != nil {
		select {
		case op = <-sc.Pool:
			op.OPConfig.Params = params
			op.OPEnv = env
			op.timeout = sc.timeout
			op.newEval = sc.newEval
			op.newParse = sc.newParse
		default:
			return nil, false
		}
	} else {
		op = &SimpleCutOP{
			OPConfig: video.OPConfig{
				Host:   sc.OPConfig.Host,
				Params: params,
			},
			OPEnv:    env,
			timeout:  sc.timeout,
			newEval:  sc.newEval,
			newParse: sc.newParse,
		}
	}
	eval := sc.eval
	if sc.newEval != nil {
		eval = sc.newEval(params)
	}
	parse := sc.parse
	if sc.newParse != nil {
		parse = sc.newParse(params)
	}
	if parse == nil {
		parse = func(v interface{}) (CutResults, bool) {
			results, ok := v.(CutResults)
			return results, ok
		}
	}
	op.eval = eval
	op.parse = parse
	return *op, true
}

func (sc SimpleCutOP) Reset(ctx context.Context) error {
	if sc.Pool != nil {
		sc.Pool <- &sc
	}
	return nil
}

func (sc SimpleCutOP) Count() int32 {
	if sc.Pool != nil {
		return int32(len(sc.Pool))
	}
	return video.MAX_OP_COUNT
}

func (sc SimpleCutOP) NewCuts(
	ctx context.Context, params, originParams *vframe.VframeParams,
) video.CutsV2 {
	return video.NewSimpleCutsV2(params, originParams,
		func(ctx context.Context, uris []string) ([]interface{}, error) {
			var (
				xl     = xlog.FromContextSafe(ctx)
				client = ahttp.NewQiniuStubRPCClient(sc.OPEnv.Uid, sc.OPEnv.Utype, sc.timeout)
				result = make([]interface{}, 0, len(uris))
			)

			xl.Infof("query : %d", len(uris))

			for _, uri := range uris {
				var resp interface{}
				var f = func(ctx context.Context) (err error) {
					resp, err = sc.eval(ctx, client, sc.OPConfig.Host, uri)
					return
				}
				if err := ahttp.CallWithRetry(
					ctx, []int{530}, []func(context.Context) error{f, f},
				); err != nil {
					xl.Errorf("query failed. error: %v, resp: %#v", err, resp)
					return nil, err //失败则返回失败
				}
				result = append(result, resp)
			}

			xl.Debugf("query done. %#v", result)
			return result, nil
		},
		func(ctx context.Context, _v interface{}) ([]string, []float32, error) {
			if _v == nil {
				return nil, nil, nil
			}
			v, ok := sc.parse(_v)
			// v, ok := _v.(CutResults)
			if !ok {
				return nil, nil, nil
			}
			var (
				n      = v.Len()
				labels = make([]string, 0, n)
				scores = make([]float32, 0, n)
			)
			for i := 0; i < n; i++ {
				_label, _score, _ok := v.Parse(i)
				if !_ok {
					continue
				}
				var found = false
				for j, label := range labels {
					if label == _label {
						if _score > scores[j] {
							scores[j] = _score
						}
						found = true
						break
					}
				}
				if found {
					continue
				}
				labels = append(labels, _label)
				scores = append(scores, _score)
			}
			return labels, scores, nil
		},
	)
}
