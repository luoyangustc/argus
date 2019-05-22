package utility

import (
	"context"
	"strings"
	"time"

	"qiniu.com/auth/authstub.v1"
)

type _EvalSceneReq struct {
	Data struct {
		URI string `json:"uri"`
	} `json:"data"`
	Params struct {
		Threshold float32 `json:"threshold,omitempty"`
		Limit     int     `json:"limit,omitempty"`
	} `json:"params,omitempty"`
}

type _EvalSceneResp struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Result  struct {
		Confidences []struct {
			Class   string   `json:"class"`
			Index   int      `json:"index"`
			Label   []string `json:"label"`
			Score   float32  `json:"score"`
			LabelCN string   `json:"label_cn,omitempty"`
		} `json:"confidences"`
	} `json:"result"`
}

type iScene interface {
	Eval(context.Context, _EvalSceneReq, _EvalEnv) (_EvalSceneResp, error)
}

type _Scene struct {
	host    string
	timeout time.Duration
}

func newScene(host string, timeout time.Duration) _Scene {
	return _Scene{host: host, timeout: timeout}
}

func (s _Scene) Eval(ctx context.Context, req _EvalSceneReq, env _EvalEnv) (resp _EvalSceneResp, err error) {
	var (
		url    = s.host + "/v1/eval/scene"
		client = newRPCClient(env, s.timeout)
	)

	err = callRetry(ctx,
		func(context.Context) error {
			return client.CallWithJson(ctx, &resp, "POST", url, req)
		})
	return
}

type SceneReq _EvalSceneReq
type SceneResp _EvalSceneResp

func (s *Service) PostScene(ctx context.Context, args *SceneReq, env *authstub.Env) (ret *SceneResp, err error) {
	defer func(begin time.Time) {
		responseTimeAtServer("scene", "").
			Observe(durationAsFloat64(time.Since(begin)))
		httpRequestsCounter("scene", "", formatError(err)).Inc()
	}(time.Now())

	var (
		ctex, xl = ctxAndLog(ctx, env.W, env.Req)
	)

	xl.Infof("scene args:%v", args)
	if strings.TrimSpace(args.Data.URI) == "" {
		return nil, ErrArgs
	}

	resp, err := s.iScene.Eval(ctex, _EvalSceneReq(*args), _EvalEnv{Uid: env.UserInfo.Uid, Utype: env.UserInfo.Utype})
	if err != nil {
		xl.Errorf("query eval scene error:", err)
		return nil, err
	}
	ret = new(SceneResp)
	*ret = SceneResp(resp)
	if env.UserInfo.Utype != NoChargeUtype {
		setStateHeader(env.W.Header(), "SCENE", 1)
	}
	return
}
