package utility

import (
	"context"
	"strings"
	"time"

	"qiniu.com/auth/authstub.v1"
)

type _EvalDetectionReq struct {
	Data struct {
		URI string `json:"uri"`
	} `json:"data"`
	Params struct {
		Threshold float32 `json:"threshold,omitempty"`
	} `json:"params,omitempty"`
}

type _EvalDetection struct {
	Index   int      `json:"index"`
	Class   string   `json:"class"`
	Score   float32  `json:"score"`
	Pts     [][2]int `json:"pts"`
	LabelCN string   `json:"label_cn,omitempty"`
}

type _EvalDetectionResp struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Result  struct {
		Detections []_EvalDetection `json:"detections"`
	} `json:"result"`
}

type iDetection interface {
	Eval(context.Context, _EvalDetectionReq, _EvalEnv) (_EvalDetectionResp, error)
}

func newObjectDetection(host string, timeout time.Duration) _Detect {
	return _Detect{host: host, timeout: timeout}
}

type _Detect struct {
	host    string
	timeout time.Duration
}

func (o _Detect) Eval(ctx context.Context, args _EvalDetectionReq, env _EvalEnv) (ret _EvalDetectionResp, err error) {
	var (
		url    = o.host + "/v1/eval/detection"
		client = newRPCClient(env, o.timeout)
	)
	err = callRetry(ctx,
		func(ctx context.Context) error {
			return client.CallWithJson(ctx, &ret, "POST", url, args)
		})
	return
}

//------------------------------------------------------------//
type DetectionReq _EvalFaceDetectReq
type DetectionResp _EvalDetectionResp

func (s *Service) PostDetect(ctx context.Context, args *DetectionReq, env *authstub.Env) (*DetectionResp, error) {
	var (
		ctex, xl = ctxAndLog(ctx, env.W, env.Req)
		_env     = _EvalEnv{Uid: env.UserInfo.Uid, Utype: env.UserInfo.Utype}
	)

	if strings.TrimSpace(args.Data.URI) == "" {
		return nil, ErrArgs
	}

	var dtReq _EvalDetectionReq
	dtReq.Data.URI = args.Data.URI
	resp, err := s.iDetection.Eval(ctex, dtReq, _env)
	if err != nil {
		xl.Errorf("call object detection error:%v", err)
		return nil, err
	}
	ret := DetectionResp(resp)
	if env.UserInfo.Utype != NoChargeUtype {
		setStateHeader(env.W.Header(), "OBJECT_DETECTION", 1)
	}
	return &ret, nil
}
