package utility

import (
	"context"
	"strings"
	"time"

	"github.com/qiniu/http/httputil.v1"
	"qiniu.com/argus/utility/evals"
	"qiniu.com/auth/authstub.v1"
)

type _BlueDetectionReq struct {
	Data struct {
		URI string `json:"uri"`
	} `json:"data"`
}

type _BlueDetectionResp struct {
	Code    int            `json:"code"`
	Message string         `json:"message"`
	Result  _BlueDetResult `json:"result"`
}

type _BlueDetResult struct {
	Detections []_BlueResultDetection `json:"detections"`
}

type _BlueResultDetection struct {
	AreaRatio float32  `json:"area_ratio"`
	Class     string   `json:"class"`
	Index     int      `json:"index"`
	Score     float32  `json:"score"`
	Pts       [][2]int `json:"pts"`
}

type _BlueDReq _BlueDetectionReq
type _BlueDResp _BlueDetectionResp

type _BlueClassfiyReq evals.PulpReq
type _BlueClassfiyResp evals.PulpResp

//--------------------------------------------------------//
type iBluedD interface {
	Eval(context.Context, *_BlueDReq, *_EvalEnv) (*_BlueDResp, error)
}
type _BluedD struct {
	host    string
	timeout time.Duration
}

func newBluedD(host string, timeout time.Duration) *_BluedD {
	return &_BluedD{host: host, timeout: timeout}
}
func (b _BluedD) Eval(ctx context.Context, req *_BlueDReq, env *_EvalEnv) (ret *_BlueDResp, err error) {
	var (
		url    = b.host + "/v1/eval/blued-d"
		client = newRPCClient(*env, b.timeout)
	)
	ret = new(_BlueDResp)
	err = callRetry(ctx,
		func(ctx context.Context) error {
			return client.CallWithJson(ctx, ret, "POST", url, req)
		})
	return
}

//-------------------------------------------------------//
type iBluedClassify interface {
	Eval(context.Context, *_BlueClassfiyReq, *_EvalEnv) (*_BlueClassfiyResp, error)
}
type _BluedClassify struct {
	host    string
	timeout time.Duration
}

func newBluedClassify(host string, timeout time.Duration) *_BluedClassify {
	return &_BluedClassify{host: host}
}

func (b _BluedClassify) Eval(ctx context.Context, req *_BlueClassfiyReq, env *_EvalEnv) (ret *_BlueClassfiyResp, err error) {
	var (
		url    = b.host + "/v1/eval/blued-c"
		client = newRPCClient(*env, b.timeout)
	)
	ret = new(_BlueClassfiyResp)
	err = callRetry(ctx,
		func(ctx context.Context) error {
			return client.CallWithJson(ctx, ret, "POST", url, req)
		})
	return
}

func (s *Service) PostBluedDetection(ctx context.Context, args *_BlueDetectionReq, env *authstub.Env) (ret *_BlueDetectionResp, err error) {

	ctx, xl := ctxAndLog(ctx, env.W, env.Req)
	var (
		uid           = env.UserInfo.Uid
		utype         = env.UserInfo.Utype
		blueDReq      _BlueDReq
		blueClassyReq _BlueClassfiyReq
	)
	xl.Infof("args: %v,threshold:%v", args, s.Config.BluedDetectThreshold)
	if strings.TrimSpace(args.Data.URI) == "" {
		xl.Error("invalid argument")
		return nil, ErrArgs
	}

	blueDReq.Data.URI = args.Data.URI
	bludDResp, err := s.iBluedD.Eval(ctx, &blueDReq, &_EvalEnv{
		Uid:   uid,
		Utype: utype,
	})
	if err != nil {
		xl.Errorf("PostBluedDetection call iBluedD.Eval error:%v", err)
		return
	}

	if len(bludDResp.Result.Detections) == 0 {
		ret = &_BlueDetectionResp{
			Result: _BlueDetResult{
				Detections: make([]_BlueResultDetection, 0),
			},
		}
		setStateHeader(env.W.Header(), "BLUED", 1)
		return
	}

	blueClassyReq.Data.URI = args.Data.URI
	blueClassyReq.Params.Limit = 1
	blueClassyResp, err := s.iBluedClassify.Eval(ctx, &blueClassyReq, &_EvalEnv{
		Uid:   uid,
		Utype: utype,
	})
	if err != nil || blueClassyResp != nil && len(blueClassyResp.Result.Confidences) == 0 {
		xl.Errorf("PostBluedDetection call iBluedD.Eval error:%v,len(Result.Confidences):%v", err, len(blueClassyResp.Result.Confidences))
		err = httputil.NewError(500, "blued classify got invalid results")
		return
	}
	if blueClassyResp.Result.Confidences[0].Index == 1 && blueClassyResp.Result.Confidences[0].Score < s.Config.BluedDetectThreshold || blueClassyResp.Result.Confidences[0].Index == 0 {
		ret = &_BlueDetectionResp{
			Result: _BlueDetResult{
				Detections: make([]_BlueResultDetection, 0),
			},
		}
		setStateHeader(env.W.Header(), "BLUED", 1)
		return

	}
	ret = (*_BlueDetectionResp)(bludDResp)
	setStateHeader(env.W.Header(), "BLUED", 1)
	return
}
