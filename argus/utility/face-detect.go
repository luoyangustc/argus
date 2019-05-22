package utility

import (
	"context"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/qiniu/xlog.v1"
	"qbox.us/errors"
	"qiniu.com/auth/authstub.v1"

	"qiniu.com/argus/argus/com/util"
	"qiniu.com/argus/utility/evals"
	"qiniu.com/argus/utility/server"
)

type _EvalFaceReq struct {
	Data struct {
		URI       string `json:"uri"`
		Attribute struct {
			Pts [][2]int `json:"pts"`
		} `json:"attribute"`
	} `json:"data"`
}
type _EvalFaceAgeReq struct {
	Data struct {
		URI       string `json:"uri"`
		Attribute struct {
			Pts [][2]int `json:"pts"`
		} `json:"attribute"`
	} `json:"data"`
	Params struct {
		Limit int `json:"limit"`
	} `json:"params,omitempty"`
}

type _EvalFaceResp struct {
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

type iFaceAge interface {
	Eval(context.Context, _EvalFaceAgeReq, _EvalEnv) (_EvalFaceResp, error)
}

type _FaceAge struct {
	evals.SimpleEval
}

func newFaceAge(host string, timeout time.Duration) iFaceAge {
	return _FaceAge{SimpleEval: evals.NewSimpleEval(host, "/v1/eval/facex-age", timeout)}
}

func (fa _FaceAge) Eval(
	ctx context.Context, req _EvalFaceAgeReq, env _EvalEnv,
) (ret _EvalFaceResp, err error) {
	req.Params.Limit = -1
	err = fa.SimpleEval.Eval(ctx, env.Uid, env.Utype, req, ret)
	return
}

func FAgePostEval(ctx context.Context, resp _EvalFaceResp) (float32, error) {

	var age float32
	for _, item := range resp.Result.Confidences {
		low, high, err := AgeStrSplit(item.Class, "-")
		if err != nil {
			return 0.0, err
		}
		age += float32((low + high)) / 2 * item.Score
	}
	return age, nil
}

func AgeStrSplit(s string, delimiter string) (int, int, error) {
	var (
		low  int
		high int
	)
	ages := strings.Split(s, delimiter)
	if len(ages) == 0 {
		return 0, 0, fmt.Errorf("age string %v split with delimiter %v error", s, delimiter)
	}
	low, err := strconv.Atoi(ages[0])
	if err != nil {
		return 0, 0, fmt.Errorf("split age string %v with delimiter %v error %v", s, delimiter, err.Error())
	}
	if len(ages) == 0 {
		high = low + 10
	}
	high, err = strconv.Atoi(ages[1])
	if err != nil {
		return 0, 0, fmt.Errorf("split age string %v with delimiter %v error %v", s, delimiter, err.Error())
	}
	return low, high, nil
}

type iFaceGender interface {
	Eval(context.Context, _EvalFaceReq, _EvalEnv) (_EvalFaceResp, error)
}

type _FaceGender struct {
	evals.SimpleEval
}

func newFaceGender(host string, timeout time.Duration) iFaceGender {
	return _FaceGender{SimpleEval: evals.NewSimpleEval(host, "/v1/eval/facex-gender", timeout)}
}

func (fa _FaceGender) Eval(
	ctx context.Context, req _EvalFaceReq, env _EvalEnv,
) (ret _EvalFaceResp, err error) {
	err = fa.SimpleEval.Eval(ctx, env.Uid, env.Utype, req, ret)
	return
}

//----------------------------------------------------------------------------//

type FaceDetectReq struct {
	Data struct {
		URI string `json:"uri"`
	} `json:"data"`
	Params struct {
		Detail bool `json:"detail,omitempty"`
	} `json:"params,omitempty"`
}

type FaceDetectResp struct {
	Code    int              `json:"code"`
	Message string           `json:"message"`
	Result  FaceDetectResult `json:"result"`
}

type FaceDetectResult struct {
	Detections []FaceDetectDetail `json:"detections"`
}

type FaceDetectDetail struct {
	BoundingBox FaceDetectBox     `json:"bounding_box"`
	Age         *FaceDetectAge    `json:"age,omitempty"`
	Gender      *FaceDetectGender `json:"gender,omitempty"`
}

type FaceDetectBox struct {
	Pts   [][2]int `json:"pts"`
	Score float32  `json:"score"`
}

type FaceDetectAge struct {
	Value float32 `json:"value"`
	Score float32 `json:"score"`
}

type FaceDetectGender struct {
	Value string  `json:"value"`
	Score float32 `json:"score"`
}

func RegisterFaceDetect() {
	var eFD, eFA, eFG = "facex-detect", "facex-age", "facex-gender"
	server.RegisterEval(eFD, func(cfg server.EvalConfig) interface{} { return evals.NewFaceDetect(cfg.Host, cfg.Timeout) })
	server.RegisterEval(eFA, func(cfg server.EvalConfig) interface{} { return newFaceAge(cfg.Host, cfg.Timeout) })
	server.RegisterEval(eFG, func(cfg server.EvalConfig) interface{} { return newFaceGender(cfg.Host, cfg.Timeout) })

	server.RegisterHandler("/face/detect", &FaceDetectSrv{eFD: eFD, eFA: eFA, eFG: eFG})
}

type FaceDetectSrv struct {
	eFD, eFA, eFG string
	FD            evals.IFaceDetect
	FA            iFaceAge
	FG            iFaceGender
}

func (s *FaceDetectSrv) Init(msg json.RawMessage, is server.IServer) interface{} {
	s.FD = is.GetEval(s.eFD).(evals.IFaceDetect)
	s.FA = is.GetEval(s.eFA).(iFaceAge)
	s.FG = is.GetEval(s.eFG).(iFaceGender)
	return s
}

func (s *FaceDetectSrv) PostFaceDetect(
	ctx context.Context, args *FaceDetectReq, env *authstub.Env,
) (ret *FaceDetectResp, err error) {

	ctx, xl := util.CtxAndLog(ctx, env.W, env.Req)
	var (
		uid   = env.UserInfo.Uid
		utype = env.UserInfo.Utype
	)
	if strings.TrimSpace(args.Data.URI) == "" {
		xl.Error("empty data.uri")
		return nil, ErrArgs
	}

	var ftReq evals.SimpleReq
	ftReq.Data.URI = args.Data.URI
	ftResp, err := s.FD.Eval(ctx, ftReq, uid, utype)
	if err != nil {
		xl.Errorf("call facex-detect error:%v", err)
		return nil, err
	}
	if ftResp.Code != 0 && ftResp.Code/100 != 2 {
		xl.Errorf("call facex-detect error:%v", ftResp)
		return nil, errors.New("call facex-detect get no zero code")
	}

	ret = new(FaceDetectResp)
	if !args.Params.Detail {
		for _, dt := range ftResp.Result.Detections {
			ret.Result.Detections = append(
				ret.Result.Detections,
				FaceDetectDetail{BoundingBox: FaceDetectBox{Pts: dt.Pts, Score: dt.Score}},
			)
		}
	} else {

		waiter := sync.WaitGroup{}
		waiter.Add(len(ftResp.Result.Detections))
		lk := new(sync.Mutex)
		ctxs := make([]context.Context, 0, len(ftResp.Result.Detections))
		for _, dt := range ftResp.Result.Detections {
			ctx2 := util.SpawnContext(ctx)
			go func(ctx context.Context, face evals.FaceDetection) {
				defer waiter.Done()
				xl := xlog.FromContextSafe(ctx)
				var (
					faReq _EvalFaceAgeReq
					fgReq _EvalFaceReq
					env   = _EvalEnv{Uid: uid, Utype: utype}
				)
				faReq.Data.URI = args.Data.URI
				faReq.Data.Attribute.Pts = face.Pts

				fgReq.Data.URI = args.Data.URI
				fgReq.Data.Attribute.Pts = face.Pts
				var (
					faResp      _EvalFaceResp
					fgResp      _EvalFaceResp
					finalAge    float32
					inerrAge    error
					inerrGender error
				)
				inwaiter := sync.WaitGroup{}
				inwaiter.Add(2)
				ctx2 := spawnContext(ctx)
				go func(ctx context.Context) {
					defer inwaiter.Done()
					faResp, inerrAge = s.FA.Eval(ctx, faReq, env)
					if inerrAge == nil && ((faResp.Code != 0 && faResp.Code/100 != 2) || len(faResp.Result.Confidences) == 0) {
						inerrAge = errors.New(fmt.Sprintf("call /v1/eval/facex-age resp :%v", faResp))
						return
					}
					if inerrAge == nil {
						finalAge, inerrAge = FAgePostEval(ctx, faResp)
					}
				}(ctx2)
				ctx3 := spawnContext(ctx)
				go func(ctx context.Context) {
					defer inwaiter.Done()
					fgResp, inerrGender = s.FG.Eval(ctx, fgReq, env)
					if inerrGender == nil && ((fgResp.Code != 0 && fgResp.Code/100 != 2) || len(fgResp.Result.Confidences) == 0) {
						inerrGender = errors.New(fmt.Sprintf("call /v1/eval/facex-gender resp:%v", fgResp))
					}
				}(ctx3)
				inwaiter.Wait()
				xl.Xput(xlog.FromContextSafe(ctx2).Xget())
				xl.Xput(xlog.FromContextSafe(ctx3).Xget())
				if inerrAge != nil || inerrGender != nil {
					xl.Errorf("fgReq:%v,query age error:%v,query gender error:%v", fgReq, inerrAge, inerrGender)
					return
				}

				lk.Lock()
				ret.Result.Detections = append(
					ret.Result.Detections,
					FaceDetectDetail{
						BoundingBox: FaceDetectBox{
							Pts:   face.Pts,
							Score: face.Score,
						},
						Age: &FaceDetectAge{
							Value: finalAge,
							Score: faResp.Result.Confidences[0].Score,
						},
						Gender: &FaceDetectGender{
							Value: fgResp.Result.Confidences[0].Class,
							Score: fgResp.Result.Confidences[0].Score,
						}},
				)
				lk.Unlock()
			}(ctx2, dt)
			ctxs = append(ctxs, ctx2)
		}
		waiter.Wait()
		for _, ctx2 := range ctxs {
			xl.Xput(xlog.FromContextSafe(ctx2).Xget())
		}
	}

	if len(ret.Result.Detections) == 0 {
		ret.Message = "No valid face info detected"
	}

	if err == nil && env.UserInfo.Utype != NoChargeUtype {
		setStateHeader(env.W.Header(), "FACE_DETECT", 1)
	}
	return
}
