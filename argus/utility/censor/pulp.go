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

type PulpReq struct {
	Data struct {
		URI string `json:"uri"`
	} `json:"data"`
	Params struct {
		Limit int `json:"limit"`
	} `json:"params,omitempty"`
}

type PulpResp struct {
	Code    int        `json:"code"`
	Message string     `json:"message"`
	Result  PulpResult `json:"result"`
}

type PulpResult struct {
	Label       int     `json:"label"`
	Score       float32 `json:"score"`
	Review      bool    `json:"review"`
	Confidences []struct {
		Index int     `json:"index"`
		Class string  `json:"class"`
		Score float32 `json:"score"`
	} `json:"confidences,omitempty"`
}

func (s *Service) postPulp(
	ctx context.Context, args *evals.PulpReq, img server.Image, uid, utype uint32,
) (ret PulpResp, err error) {

	var (
		xl = xlog.FromContextSafe(ctx)
	)

	var (
		limit = args.Params.Limit
		eResp evals.PulpResp
	)
	args.Params.Limit = 3

	_t1 := time.Now()
	eResp, err = s.ePulp.Eval(ctx, *args, uid, utype)
	server.ResponseTimeAtClient("pulp", "serving_pulp", "").Observe(server.DurationAsFloat64(time.Since(_t1)))

	if err != nil || (eResp.Code != 0 && eResp.Code/100 != 2) || len(eResp.Result.Confidences) != 3 {
		xl.Errorf("query pulp error:%v,resp:%v", err, eResp)
		// err = errors.New("got invalid results from serving-pulp")
		return
	}
	

	for i, cf := range eResp.Result.Confidences {
		if cf.Score >= s.Config.PulpReviewThreshold {
			eResp.Result.Confidences[i].Score = 0.4*(cf.Score-s.Config.PulpReviewThreshold)/(1-s.Config.PulpReviewThreshold) + 0.6
		} else {
			eResp.Result.Confidences[i].Score = 0.6 * (cf.Score / s.Config.PulpReviewThreshold)
		}
	}

	ret.Result.Label = eResp.Result.Confidences[0].Index
	ret.Result.Score = eResp.Result.Confidences[0].Score
	if eResp.Result.Confidences[0].Score < 0.6 {
		ret.Result.Review = true
	}

	if limit > 1 {
		if limit > 3 {
			limit = 3
		}
		ret.Result.Confidences = eResp.Result.Confidences[:limit]
	}

	return
}

func (s *Service) PostPulp(ctx context.Context, args *PulpReq, env *authstub.Env) (*PulpResp, error) {

	var err error

	defer func(begin time.Time) {
		server.ResponseTimeAtServer("pulp", "").
			Observe(server.DurationAsFloat64(time.Since(begin)))
		server.HttpRequestsCounter("pulp", "", server.FormatError(err)).Inc()
	}(time.Now())

	ctx, xl := util.CtxAndLog(ctx, env.W, env.Req)
	if strings.TrimSpace(args.Data.URI) == "" {
		err = server.ErrArgs
		return nil, err
	}

	uri, _ := server.ImproveURI(args.Data.URI, env.Uid)
	img, err := s.ParseImage(ctx, uri)
	if err != nil && err != image.ErrFormat {
		xl.Infof("parse image failed. %v", err)
		return nil, err
	}

	var (
		req evals.PulpReq
	)
	req.Data.URI = args.Data.URI
	if img.URI != nil {
		req.Data.URI = *img.URI
	}
	req.Params.Limit = args.Params.Limit
	ret, err := s.postPulp(ctx, &req, img, env.Uid, env.Utype)

	if err == nil && env.UserInfo.Utype != server.NoChargeUtype {
		if ret.Result.Review {
			server.SetStateHeader(env.W.Header(), "PULP-Depend", 1)
		} else {
			server.SetStateHeader(env.W.Header(), "PULP-Certain", 1)
		}
	}
	return &ret, err
}
