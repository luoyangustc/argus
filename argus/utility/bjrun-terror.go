package utility

import (
	"context"
	"fmt"
	"strings"

	"github.com/qiniu/http/httputil.v1"
	"qiniu.com/auth/authstub.v1"

	"qiniu.com/argus/utility/evals"
)

type BjRTerrorReq struct {
	Data struct {
		URI string `json:"uri"`
	} `json:"data"`
}
type BjRTerrorResp struct {
	Code    int             `json:"code"`
	Message string          `json:"message"`
	Result  BjRTerrorResult `json:"result"`
}

type BjRTerrorResult struct {
	Label  int     `json:"label"`
	Score  float32 `json:"score"`
	Class  string  `json:"class"`
	Review bool    `json:"review"`
}

var BjRClassMap = map[string]string{
	"knives":       "刀具",
	"guns":         "枪械",
	"islamic flag": "星月旗",
	"tibetan flag": "雪山狮子旗",
	"isis flag":    "ISIS旗帜",
	"march":        "游行集会",
	"bloodiness":   "血腥场景",
	"bomb":         "爆炸场景",
	"beheaded":     "行刑斩首",
	"fight":        "打架斗殴",
	"normal":       "正常",
}

func (t *BjRTerrorResp) _Final(threshold float32) {
	if t.Result.Score < threshold {
		t.Result.Review = true
	}
	if _, ok := BjRClassMap[t.Result.Class]; ok {
		t.Result.Class = BjRClassMap[t.Result.Class]
	}
}

func (s *Service) PostBjrunTerror(ctx context.Context, args *BjRTerrorReq, env *authstub.Env) (ret *BjRTerrorResp, err error) {

	var (
		ctex, xl        = ctxAndLog(ctx, env.W, env.Req)
		reviewThreshold = s.BjRTerrorThreshold[1]
		detectThreshold = s.BjRTerrorThreshold[0]
	)
	if strings.TrimSpace(args.Data.URI) == "" {
		xl.Error("empty data.uri")
		return nil, ErrArgs
	}
	ret = new(BjRTerrorResp)

	var (
		tdReq         evals.SimpleReq
		tdResp        evals.TerrorDetectResp
		maxScore      float32
		maxScoreIndex int
	)
	maxScoreIndex = -1
	tdReq.Data.URI = args.Data.URI

	tdResp, tderr := s.terrorDetect.Eval(ctex, tdReq, env.Uid, env.Utype)

	if tderr == nil && tdResp.Code != 0 && tdResp.Code/100 != 2 {
		xl.Errorf("call /v1/eval/terror-detect error resp :%v", tdResp)
		retCode := StatusUnkownErrCode
		if tdResp.Code >= 300 {
			retCode = tdResp.Code
		}
		tderr = httputil.NewError(retCode, fmt.Sprintf("call terror-detect error :%v", tdResp.Message))
	}

	for i, d := range tdResp.Result.Detections {
		if d.Score > maxScore {
			maxScoreIndex = i
			maxScore = d.Score
		}
	}
	if maxScore > detectThreshold {
		ret.Result.Label = 1
		ret.Result.Score = maxScore
		ret.Result.Review = false
		ret.Result.Class = tdResp.Result.Detections[maxScoreIndex].Class
		ret._Final(reviewThreshold)
		ret.Message = "success"

		if ret.Result.Review {
			setStateHeader(env.W.Header(), "TERROR-Depend", 1)
		} else {
			setStateHeader(env.W.Header(), "TERROR-Certain", 1)
		}
		return
	}

	xl.Infof("call /v1/eval/terror-detect no target with rate larger than %v detected :%v", s.Config.TerrorThreshold, tdResp)

	var (
		tfReq  evals.SimpleReq
		tfResp evals.TerrorClassifyResp
	)
	tfReq.Data.URI = args.Data.URI
	tfResp, tferr := s.terrorClassify.Eval(ctex, tfReq, env.Uid, env.Utype)
	if tferr != nil || (tfResp.Code != 0 && tfResp.Code/100 != 2) || len(tfResp.Result.Confidences) == 0 {
		xl.Errorf("call /v1/eval/terror-clsssify error resp :%v", tfResp)
		if maxScoreIndex != -1 {
			ret.Result.Label = 1
			ret.Result.Score = tdResp.Result.Detections[maxScoreIndex].Score
			ret.Result.Review = false
			ret.Result.Class = tdResp.Result.Detections[maxScoreIndex].Class
			ret._Final(reviewThreshold)
			ret.Message = "success"

			if ret.Result.Review {
				setStateHeader(env.W.Header(), "TERROR-Depend", 1)
			} else {
				setStateHeader(env.W.Header(), "TERROR-Certain", 1)
			}

			return
		} else if tderr == nil {
			ret.Result.Label = 0
			ret.Result.Score = 1.0
			ret.Result.Class = "normal"
			ret._Final(reviewThreshold)
			ret.Message = "success"

			if ret.Result.Review {
				setStateHeader(env.W.Header(), "TERROR-Depend", 1)
			} else {
				setStateHeader(env.W.Header(), "TERROR-Certain", 1)
			}

			return
		}
		if tferr != nil { //直接返回服务端的错误内容
			xl.Errorf("both of classify and detect service failed,classify error:%v, detect error:%v", tferr, tderr)
			return nil, tferr
		}
		return nil, tderr
	}

	ret.Result.Label = 1
	ret.Message = "success"
	ret.Result.Score = tfResp.Result.Confidences[0].Score
	ret.Result.Class = tfResp.Result.Confidences[0].Class
	ret.Result.Review = false
	if tfResp.Result.Confidences[0].Index == -1 || tfResp.Result.Confidences[0].Index >= 11 && tfResp.Result.Confidences[0].Index <= 32 {
		ret.Result.Label = 0
	}
	ret._Final(reviewThreshold)

	if ret.Result.Review {
		setStateHeader(env.W.Header(), "TERROR-Depend", 1)
	} else {
		setStateHeader(env.W.Header(), "TERROR-Certain", 1)
	}

	return
}
