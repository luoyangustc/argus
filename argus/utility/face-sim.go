package utility

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"math"
	"sync"

	"github.com/qiniu/xlog.v1"
	"qbox.us/errors"
	"qiniu.com/auth/authstub.v1"

	"qiniu.com/argus/argus/com/util"
	"qiniu.com/argus/utility/evals"
	"qiniu.com/argus/utility/server"
)

type FaceSimReq struct {
	Data []struct {
		URI string `json:"uri"`
	} `json:"data"`
}

type FaceSimResp struct {
	Code    int           `json:"code"`
	Message string        `json:"message"`
	Result  FaceSimResult `json:"result"`
}

type FaceSimResult struct {
	Faces      []FaceSimBox `json:"faces"`
	Similarity float32      `json:"similarity"`
	Same       bool         `json:"same"`
}

type FaceSimBox struct {
	Pts   [][2]int `json:"pts"`
	Score float32  `json:"score"`
}

func LargestFace(Detections []evals.FaceDetection) int {
	var (
		lgFaceArea  int
		lgFaceIndex int
	)
	for i, dt := range Detections {
		area := (dt.Pts[1][0] - dt.Pts[0][0]) * (dt.Pts[2][1] - dt.Pts[1][1])
		if area > lgFaceArea {
			lgFaceArea = area
			lgFaceIndex = i
		}
	}
	return lgFaceIndex
}

func RegisterFaceSim() {
	var eFD, eFF = "facex-detect", "facex-feature-v3"
	server.RegisterEval(eFD, func(cfg server.EvalConfig) interface{} { return evals.NewFaceDetect(cfg.Host, cfg.Timeout) })
	server.RegisterEval(eFF, func(cfg server.EvalConfig) interface{} { return evals.NewFaceFeature(cfg.Host, cfg.Timeout, "-v3") })

	server.RegisterHandler("/face/sim", &FaceSimSrv{eFD: eFD, eFF: eFF})
}

type FaceSimSrv struct {
	threshold float32

	eFD, eFF string
	FD       evals.IFaceDetect
	FF       evals.IFaceFeature
}

func (s *FaceSimSrv) Init(msg json.RawMessage, is server.IServer) interface{} {
	var cfg = struct {
		Threshold float32 `json:"threshold"`
	}{}
	_ = json.Unmarshal(msg, &cfg)
	s.threshold = cfg.Threshold

	s.FD = is.GetEval(s.eFD).(evals.IFaceDetect)
	s.FF = is.GetEval(s.eFF).(evals.IFaceFeature)
	return s
}

func (s *FaceSimSrv) PostFaceSim(
	ctx context.Context, args *FaceSimReq, env *authstub.Env,
) (ret *FaceSimResp, err error) {

	ctx, xl := util.CtxAndLog(ctx, env.W, env.Req)
	var (
		uid   = env.UserInfo.Uid
		utype = env.UserInfo.Utype
	)
	if len(args.Data) != 2 {
		xl.Error("invalid number of uris")
		return nil, ErrArgs
	}

	faces := make([]FaceSimBox, len(args.Data))
	features := make([][]byte, len(args.Data))
	waiter := sync.WaitGroup{}
	waiter.Add(len(args.Data))
	ctxs := make([]context.Context, 0, 2)
	for i, dt := range args.Data {
		ctx2 := util.SpawnContext(ctx)
		go func(ctx context.Context, uri string, ord int) {
			defer waiter.Done()
			xl := xlog.FromContextSafe(ctx)
			var (
				fdReq evals.SimpleReq
				ftReq evals.FaceReq
			)
			fdReq.Data.URI = uri
			fdResp, _err := s.FD.Eval(ctx, fdReq, uid, utype)
			if _err != nil {
				xl.Errorf("call facex-detect error:%v，", _err)
				return
			}
			if (fdResp.Code != 0 && fdResp.Code/100 != 2) || len(fdResp.Result.Detections) == 0 {
				xl.Errorf("call facex-detect error:%v", fdResp)
				return
			}

			//TODO Select the largest face,not the one with largest score?
			ftReq.Data.URI = uri
			LgFaceIndex := LargestFace(fdResp.Result.Detections)
			ftReq.Data.Attribute.Pts = fdResp.Result.Detections[LgFaceIndex].Pts
			ftResp, _err := s.FF.Eval(ctx, ftReq, uid, utype)
			if _err != nil {
				xl.Errorf("call facex-feature error:%v，uri:%v", _err, uri)
				return
			}
			if len(ftResp) < 5 {
				xl.Errorf("feature length error:%v, uri:%v", ftResp, uri)
				return
			}

			if fdResp.Result.Detections[LgFaceIndex].Score > 1.0 {
				fdResp.Result.Detections[LgFaceIndex].Score = 1.0
			}
			features[ord] = ftResp
			faces[ord] = FaceSimBox{
				Pts:   fdResp.Result.Detections[LgFaceIndex].Pts,
				Score: fdResp.Result.Detections[LgFaceIndex].Score,
			}

		}(ctx2, dt.URI, i)
		ctxs = append(ctxs, ctx2)
	}
	waiter.Wait()
	for _, ctx2 := range ctxs {
		xl.Xput(xlog.FromContextSafe(ctx2).Xget())
	}

	for _, ft := range features {
		if len(ft) == 0 {
			err = errors.New("No valid features were obtained")
			return
		}
	}

	var score float32
	for i, n := 0, len(features[0]); i < n; i += 4 {
		score += math.Float32frombits(binary.BigEndian.Uint32(features[0][i:i+4])) *
			math.Float32frombits(binary.BigEndian.Uint32(features[1][i:i+4]))
	}

	var same bool
	if score >= s.threshold {
		same = true
	}

	ret = new(FaceSimResp)
	ret.Message = "success"
	ret.Result.Faces = []FaceSimBox{faces[0], faces[1]}
	ret.Result.Similarity = score
	ret.Result.Same = same
	if ret.Result.Similarity > 1.0 {
		ret.Result.Similarity = 1.0
	}
	if ret.Result.Similarity < 0 {
		ret.Result.Similarity = 0
	}

	if err == nil && env.UserInfo.Utype != NoChargeUtype {
		setStateHeader(env.W.Header(), "FACE_SIM", 1)
	}

	return
}
