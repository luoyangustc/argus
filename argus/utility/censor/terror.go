package censor

import (
	"context"
	"image"
	"strings"
	"time"

	"github.com/qiniu/xlog.v1"
	"qiniu.com/auth/authstub.v1"

	"qiniu.com/argus/argus/com/util"
	"qiniu.com/argus/utility/evals"
	"qiniu.com/argus/utility/server"
)

type TerrorReq struct {
	Data struct {
		URI string `json:"uri"`
	} `json:"data"`
	Params struct {
		Detail bool `json:"detail"`
	} `json:"params"`
}
type TerrorResp struct {
	Code    int          `json:"code"`
	Message string       `json:"message"`
	Result  TerrorResult `json:"result"`
}

type TerrorResult struct {
	Label  int     `json:"label"`
	Class  string  `json:"class,omitempty"`
	Score  float32 `json:"score"`
	Review bool    `json:"review"`
}

func (t *TerrorResp) _Final(threshold float32) {
	if t.Result.Score < threshold {
		t.Result.Review = true
	}
}

func (s *Service) postTerror(
	ctx context.Context, req evals.SimpleReq, uid, utype uint32, detail bool,
) (ret TerrorResp, err error) {

	var (
		xl     = xlog.FromContextSafe(ctx)
		pdResp evals.TerrorPreDetectResp
	)

	_t1 := time.Now()
	pdResp, pderr := s.eTerrorPreDet.Eval(ctx, req, uid, utype)
	server.ResponseTimeAtClient("terror", "terror_detect", "").
		Observe(server.DurationAsFloat64(time.Since(_t1)))
	if pderr != nil {
		xl.Errorf("query terror predetect error:%v", pderr)
	}

	var (
		dResp         evals.TerrorDetectResp
		maxScore      float32
		derr          error
		maxScoreIndex = -1
	)

	if pderr != nil || len(pdResp.Result.Detections) != 0 {
		_t2 := time.Now()
		dResp, derr = s.eTerrorDet.Eval(ctx, req, uid, utype)
		server.ResponseTimeAtClient("terror", "terror_detect", "").
			Observe(server.DurationAsFloat64(time.Since(_t2)))
		if derr != nil {
			xl.Errorf("call /v1/eval/terror-detect error resp : %v", derr)
			err = derr
			return
		}
		// if tderr == nil && tdResp.Code != 0 && tdResp.Code/100 != 2 {
		// 	xl.Errorf("call /v1/eval/terror-detect error resp :%v", tdResp)
		// 	retCode := StatusUnkownErrCode
		// 	if tdResp.Code >= 300 {
		// 		retCode = tdResp.Code
		// 	}
		// 	tderr = httputil.NewError(retCode, fmt.Sprintf("call terror-detect error :%v", tdResp.Message))
		// }
		server.HttpRequestsCounter("terror", "terror-detect", server.FormatError(derr)).Inc()
		for i, d := range dResp.Result.Detections {
			if d.Score > maxScore {
				maxScoreIndex = i
				maxScore = d.Score
			}
		}
		if maxScore > s.Config.TerrorThreshold {
			ret.Result.Label = 1
			ret.Result.Score = maxScore
			ret.Result.Review = false
			ret._Final(s.Config.TerrorThreshold)
			if detail {
				ret.Result.Class = dResp.Result.Detections[maxScoreIndex].Class
			}
			ret.Message = "success"

			return
		}
	}

	xl.Infof("call /v1/eval/terror-detect no target with rate larger than %v detected :%v",
		s.Config.TerrorThreshold, dResp)

	var (
		cResp evals.TerrorClassifyResp
	)
	_t3 := time.Now()
	cResp, cerr := s.eTerrorClassify.Eval(ctx, req, uid, utype)
	server.ResponseTimeAtClient("terror", "terror_classify", "").
		Observe(server.DurationAsFloat64(time.Since(_t3)))
	server.HttpRequestsCounter("terror", "terror-classify", server.FormatError(cerr)).Inc()
	if cerr != nil || (cResp.Code != 0 && cResp.Code/100 != 2) || len(cResp.Result.Confidences) == 0 {
		xl.Errorf("call /v1/eval/terror-clsssify error resp :%v", cResp)
		if maxScoreIndex != -1 {
			ret.Result.Label = 1
			ret.Result.Score = dResp.Result.Detections[maxScoreIndex].Score
			ret.Result.Review = false
			ret._Final(s.Config.TerrorThreshold)
			if detail {
				ret.Result.Class = dResp.Result.Detections[maxScoreIndex].Class
			}
			ret.Message = "success"

			return
		} else if derr == nil {
			ret.Result.Label = 0
			ret.Result.Score = 1.0
			ret._Final(s.Config.TerrorThreshold)
			ret.Message = "success"

			return
		}
		if cerr != nil { //直接返回服务端的错误内容
			xl.Errorf(
				"both of classify and detect service failed,classify error:%v, detect error:%v",
				cerr, derr)
			err = cerr
			return
		}
		err = derr
		return
	}

	ret.Result.Label = 1
	ret.Message = "success"
	ret.Result.Score = cResp.Result.Confidences[0].Score
	ret.Result.Review = false
	cRespClass := strings.TrimSpace(cResp.Result.Confidences[0].Class)
	if cRespClass == "normal" {
		ret.Result.Label = 0
	}
	if cResp.Result.Confidences[0].Index == -1 {
		ret._Final(s.Config.TerrorThreshold)
	}
	if detail && ret.Result.Label != 0 {
		ret.Result.Class = cResp.Result.Confidences[0].Class
	}
	if cRespClass == "bloodiness" || cRespClass == "bomb" ||
		cRespClass == "march" || // 兼容老模型
		cRespClass == "march_banner" || cRespClass == "march_crowed" {
		_t4 := time.Now()
		ptResp, pterr := s.eTerrorPostDet.Eval(ctx, req, uid, utype)
		server.ResponseTimeAtClient("terror", "terror_detect", "").
			Observe(server.DurationAsFloat64(time.Since(_t4)))
		if pderr != nil {
			xl.Errorf("query terror post detect error:%v", pterr)
			return
		}
		if len(ptResp.Result.Detections) == 0 {
			ret.Result.Review = true
		}

	}

	return
}

func (s *Service) PostTerror(
	ctx context.Context, args *TerrorReq, env *authstub.Env,
) (*TerrorResp, error) {

	var err error
	defer func(begin time.Time) {
		server.ResponseTimeAtServer("terror", "").
			Observe(server.DurationAsFloat64(time.Since(begin)))
		server.HttpRequestsCounter("terror", "", server.FormatError(err)).Inc()
	}(time.Now())

	ctx, xl := util.CtxAndLog(ctx, env.W, env.Req)
	if strings.TrimSpace(args.Data.URI) == "" {
		xl.Error("empty data.uri")
		return nil, server.ErrArgs
	}
	uri, _ := server.ImproveURI(args.Data.URI, env.Uid)
	img, err := s.ParseImage(ctx, uri)
	if err != nil && err != image.ErrFormat {
		xl.Infof("parse image failed. %v", err)
		return nil, err
	}

	var req evals.SimpleReq
	req.Data.URI = args.Data.URI
	if img.URI != nil {
		req.Data.URI = *img.URI
	}
	ret, err := s.postTerror(ctx, req, env.Uid, env.Utype, args.Params.Detail)

	if err == nil && env.UserInfo.Utype != server.NoChargeUtype {
		if ret.Result.Review {
			server.SetStateHeader(env.W.Header(), "TERROR-Depend", 1)
		} else {
			server.SetStateHeader(env.W.Header(), "TERROR-Certain", 1)
		}
	}

	return &ret, err
}
