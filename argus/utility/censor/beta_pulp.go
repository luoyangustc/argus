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

func (s *Service) postBetaPulp(
	ctx context.Context, args *evals.PulpReq, img server.Image, uid, utype uint32,
) (ret PulpResp, err error) {

	var (
		xl      = xlog.FromContextSafe(ctx)
		eResp   evals.PulpResp
		pdetReq evals.PulpDetectReq
	)
	args.Params.Limit = 3

	_t1 := time.Now()
	eResp, err = s.ePulp.Eval(ctx, *args, uid, utype)
	server.ResponseTimeAtClient("beta_pulp", "serving_pulp", "").Observe(server.DurationAsFloat64(time.Since(_t1)))

	if err != nil || (eResp.Code != 0 && eResp.Code/100 != 2) || len(eResp.Result.Confidences) != 3 {
		xl.Errorf("query pulp error:%v,resp:%v", err, eResp)
		// err = errors.New("got invalid results from serving-pulp")
		return
	}

	pdetReq.Data.URI = args.Data.URI
	_t2 := time.Now()
	dResp, err := s.ePulpDetect.Eval(ctx, pdetReq, uid, utype)
	server.ResponseTimeAtClient("beta_pulp", "serving_pulp_detect", "").Observe(server.DurationAsFloat64(time.Since(_t2)))

	if err != nil || (dResp.Code != 0 && dResp.Code/100 != 2) {
		xl.Errorf("query pulp-detect error:%v,resp:%v", err, dResp)
		return
	}

	if len(dResp.Result.Detections) != 0 && eResp.Result.Confidences[0].Index != 0 {
		maxSocre := dResp.Result.Detections[0].Score
		for _, dt := range dResp.Result.Detections {
			if dt.Score > maxSocre {
				maxSocre = dt.Score
			}
		}
		ret.Result.Score = (maxSocre - 0.6) * 1.5 //new_score = (old_score - threshold)*0.6 / (1-threshold)
		ret.Result.Label = 0
		ret.Result.Review = true
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

	return
}

func (s *Service) PostBetaPulp(ctx context.Context, args *PulpReq, env *authstub.Env) (*PulpResp, error) {

	var err error

	defer func(begin time.Time) {
		server.ResponseTimeAtServer("beta-pulp", "").
			Observe(server.DurationAsFloat64(time.Since(begin)))
		server.HttpRequestsCounter("beta-pulp", "", server.FormatError(err)).Inc()
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
	ret, err := s.postBetaPulp(ctx, &req, img, env.Uid, env.Utype)

	if err == nil && env.UserInfo.Utype != server.NoChargeUtype {
		if ret.Result.Review {
			server.SetStateHeader(env.W.Header(), "BETA-PULP-Depend", 1)
		} else {
			server.SetStateHeader(env.W.Header(), "BETA-PULP-Certain", 1)
		}
	}
	return &ret, err
}
