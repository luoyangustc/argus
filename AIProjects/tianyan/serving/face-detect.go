package serving

import (
	"context"
	"time"

	"github.com/qiniu/rpc.v3"
)

type EvalFaceDetectReq struct {
	Data struct {
		URI string `json:"uri"`
	} `json:"data"`
}

type EvalFaceDetection struct {
	Index int      `json:"index"`
	Class string   `json:"class"`
	Score float32  `json:"score"`
	Pts   [][2]int `json:"pts"`
}

type EvalFaceDetectResp struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Result  struct {
		Detections []EvalFaceDetection `json:"detections"`
	} `json:"result"`
}

type FaceDetect interface {
	Eval(context.Context, EvalFaceDetectReq) (EvalFaceDetectResp, error)
}

type _FaceDetect struct {
	url     string
	timeout time.Duration
	*rpc.Client
}

func NewFaceDetect(conf EvalConfig) FaceDetect {
	url := conf.Host + "/v1/eval/facex-detect"
	if conf.URL != "" {
		url = conf.URL
	}
	return _FaceDetect{url: url, timeout: time.Duration(conf.Timeout) * time.Second}
}

func (fd _FaceDetect) Eval(
	ctx context.Context, req EvalFaceDetectReq,
) (EvalFaceDetectResp, error) {

	var (
		client *rpc.Client
		resp   EvalFaceDetectResp
	)
	if fd.Client == nil {
		client = NewDefaultStubRPCClient(fd.timeout)
	} else {
		client = fd.Client
	}
	err := callRetry(ctx,
		func(ctx context.Context) error {
			return client.CallWithJson(ctx, &resp, "POST", fd.url, &req)
		})
	return resp, err
}
