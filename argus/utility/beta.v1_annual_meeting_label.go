package utility

import (
	"context"
	"strings"

	"qbox.us/errors"
	"qiniu.com/auth/authstub.v1"
)

//----------------------------------------------------------------------//
type HumanLabelReq FaceDetectReq

type HumanLabelResp struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Result  struct {
		Tags []string `json:"tags"`
	} `json:"result"`
}

var _LabelConvertor = map[string]string{"__background__": "校花校草我爱你", "beard": "#脏茬#", "black_frame_glasses": "学霸、教授", "police_cap": "魅力新一代", "sun_glasses": "Cool Guy", "stud_earrings": "我最闪亮", "mouth_mask": "明星脸", "bangs": "校花校草我爱你", "tattoo": "校花校草我爱你", "shirt": "交际花", "suit": "学生会主席", "tie": "学生会主席", "belt": "学生会主席", "jeans": "低调精英", "shorts": "美腿控", "leg_hair": "美腿控", "military_uniform": "校花校草我爱你", "under_shirt": "班花班草我爱你", "gloves": "班花班草我爱你", "pecs": "班花班草我爱你", "abdominal_muscles": "Cool Guy", "calf": "美腿控", "briefs": "班花班草我爱你", "boxers": "班花班草我爱你", "butt": "最美的翘臀", "leather_shoes": "学生会主席", "black_socks": "学生会主席", "white_socks": "班花班草我爱你", "feet": "魅力新一代", "non_leather_shoes": "魅力新一代", "hot_pants": "最美的翘臀"}

func (s *Service) PostBetaHumanLabel(ctx context.Context, args *HumanLabelReq, env *authstub.Env) (ret *HumanLabelResp, err error) {

	ctx, xl := ctxAndLog(ctx, env.W, env.Req)
	var (
		uid   = env.UserInfo.Uid
		utype = env.UserInfo.Utype
	)

	if strings.TrimSpace(args.Data.URI) == "" {
		xl.Error("empty data.uri")
		return nil, ErrArgs
	}

	var ftReq _EvalFaceDetectReq
	ftReq.Data.URI = args.Data.URI
	ftResp, err := s.iFaceDetect.Eval(ctx, ftReq, _EvalEnv{Uid: uid, Utype: utype})
	if err != nil {
		xl.Errorf("call facex-detect error:%v", err)
		return nil, err
	}
	if ftResp.Code != 0 && ftResp.Code/100 != 2 {
		xl.Errorf("call facex-detect error:%v", ftResp)
		return nil, errors.New("call facex-detect get no zero code")
	}

	ret = new(HumanLabelResp)
	if len(ftResp.Result.Detections) >= 5 {
		ret.Result.Tags = append(ret.Result.Tags, "we are 伐木累")
		return
	}

	var bluedRep _BlueDReq
	bluedRep.Data.URI = args.Data.URI
	bluedResp, err := s.iBluedD.Eval(ctx, &bluedRep, &_EvalEnv{Uid: uid, Utype: utype})
	if err != nil {
		xl.Errorf("call iblued error:%v", ftResp)
		return nil, errors.New("call iblued get no zero code")
	}

	var temp = make(map[string]int)
	for _, dt := range bluedResp.Result.Detections {
		if v, ok := _LabelConvertor[dt.Class]; ok {
			if _, _ok := temp[v]; _ok {
				continue
			}
			ret.Result.Tags = append(ret.Result.Tags, v)
			temp[v] = 1
		} else {
			xl.Errorf("unknown class:%v", dt)
		}
	}

	if len(ret.Result.Tags) == 0 {
		return nil, errors.New("call no available label for the image")
	}

	return
}
