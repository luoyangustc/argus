package censor

import (
	"context"
	"encoding/base64"
	"image"
	"strings"
	"sync"

	"github.com/qiniu/xlog.v1"
	util "qiniu.com/argus/argus/com/util"
	"qiniu.com/argus/utility"
	"qiniu.com/argus/utility/evals"
	"qiniu.com/argus/utility/server"
	"qiniu.com/auth/authstub.v1"
)

// FaceSearchV2Req ...
type FaceSearchV2Req struct {
	Data struct {
		URI string `json:"uri"`
	} `json:"data"`
	Params struct {
		Limit int `json:"limit,omitempty"`
	} `json:"params,omitempty"`
}

// FaceSearchV2Resp ...
type FaceSearchV2Resp struct {
	Code    int                `json:"code"`
	Message string             `json:"message"`
	Result  FaceSearchV2Result `json:"result"`
}

// FaceSearchV2Result ...
type FaceSearchV2Result struct {
	Review     bool                 `json:"review"`
	Detections []FaceSearchV2Detail `json:"detections"`
}

// FaceSearchV2Detail ...
type FaceSearchV2Detail struct {
	BoundingBox utility.FaceDetectBox `json:"boundingBox"`
	Politician  []Pvalue              `json:"politician"`
}

type Pvalue struct {
	Name   string                  `json:"name,omitempty"`
	Group  string                  `json:"group,omitempty"`
	Score  float32                 `json:"score"`
	Review bool                    `json:"review"`
	Sample *FaceSearchDetailSample `json:"sample,omitempty"`
}

func (s *Service) postSearchPolitician(
	ctx context.Context, req FaceSearchV2Req, uid, utype uint32,
) (ret FaceSearchV2Resp, err error) {

	var (
		xl = xlog.FromContextSafe(ctx)
	)

	var fdreq evals.SimpleReq
	fdreq.Data.URI = req.Data.URI
	dResp, err := s.eFaceDet.Eval(ctx, fdreq, uid, utype)
	if err != nil {
		xl.Errorf("call facex-detect error: %s %v", req.Data.URI, err)
		return
	}
	if dResp.Code != 0 && dResp.Code/100 != 2 {
		xl.Errorf("call facex-detect failed: %s %d %s", req.Data.URI, dResp.Code, dResp.Message)
		// return nil, errors.New("call facex-detect failed")
		return
	}

	var (
		waiter = sync.WaitGroup{}
		lock   = new(sync.Mutex)
	)
	ret.Result.Detections = make([]FaceSearchV2Detail, 0, len(dResp.Result.Detections))
	ctxs := make([]context.Context, 0, len(dResp.Result.Detections))
	for _, d := range dResp.Result.Detections {
		waiter.Add(1)
		ctx2 := util.SpawnContext(ctx)
		// go func(ctx context.Context, face _EvalFaceDetection) {
		func(ctx context.Context, face evals.FaceDetection) {
			defer waiter.Done()
			xl := xlog.FromContextSafe(ctx)

			var fReq evals.FaceReq
			fReq.Data.URI = req.Data.URI
			fReq.Data.Attribute.Pts = face.Pts
			ff, err1 := s.ePoliticianFFeature.Eval(ctx, fReq, uid, utype)
			if err1 != nil {
				xl.Errorf("get face feature failed. %v", err1)
				lock.Lock()
				if err == nil {
					err = err1
				}
				lock.Unlock()
				return
			}

			var mReq FaceSearchV2Req
			var confs []struct {
				Index  int     `json:"index"`
				Class  string  `json:"class"`
				Group  string  `json:"group"`
				Score  float32 `json:"score"`
				Sample struct {
					URL string   `json:"url"`
					Pts [][2]int `json:"pts"`
					ID  string   `json:"id"`
				} `json:"sample"`
			}
			mReq.Data.URI = "data:application/octet-stream;base64," +
				base64.StdEncoding.EncodeToString(ff)
			mReq.Params.Limit = req.Params.Limit
			mResp, err1 := s.ePolitician.Eval(ctx, mReq, uid, utype)
			if err1 != nil {
				xl.Errorf("get face match failed. %v", err1)
				lock.Lock()
				if err == nil {
					err = err1
				}
				lock.Unlock()
				return
			}
			if fmResp, ok := mResp.(evals.FaceSearchRespV2); ok {
				if len(fmResp.Result.Confidences) == 0 {
					xl.Errorf("unexpected s.politician.Eval result, len(mResp.Result.Confidences):%v", len(fmResp.Result.Confidences))
					return
				}
				confs = append(confs, fmResp.Result.Confidences...)
			} else if fmResp, ok := mResp.(evals.FaceSearchResp); ok {
				if fmResp.Result.Class == "" {
					xl.Errorf("unexpected s.politician.Eval result, fmResp:%v", fmResp)
					return
				}
				confs = append(confs, fmResp.Result)
			}

			detail := FaceSearchV2Detail{
				BoundingBox: utility.FaceDetectBox{
					Pts:   face.Pts,
					Score: face.Score,
				},
			}

			for _, res := range confs {
				if res.Class == "" {
					continue
				}
				var polt Pvalue
				polt.Name = res.Class
				polt.Group = res.Group
				polt.Sample = &FaceSearchDetailSample{
					URL: res.Sample.URL,
					Pts: res.Sample.Pts,
				}
				polt.Score = res.Score
				detail.Politician = append(detail.Politician, polt)
			}
			lock.Lock()
			defer lock.Unlock()
			ret.Result.Detections = append(ret.Result.Detections, detail)
		}(ctx2, d)
		ctxs = append(ctxs, ctx2)
	}
	waiter.Wait()
	for _, ctx2 := range ctxs {
		xl.Xput(xlog.FromContextSafe(ctx2).Xget())
	}

	if err != nil {
		xl.Errorf("run palitician search failed. %v", err)
		return
	}

	if len(ret.Result.Detections) == 0 {
		ret.Message = "No valid face info detected"
	}

	return
}

func (s *Service) PostSearchPolitician(
	ctx context.Context, args *FaceSearchV2Req, env *authstub.Env,
) (*FaceSearchV2Resp, error) {

	ctx, xl := util.CtxAndLog(ctx, env.W, env.Req)

	if strings.TrimSpace(args.Data.URI) == "" {
		xl.Error("empty data.uri")
		return nil, utility.ErrArgs
	}

	xl.Infof("PostSearchPolitician req args:%v", args)

	uri, _ := server.ImproveURI(args.Data.URI, env.Uid)
	img, err := s.ParseImage(ctx, uri)
	if err != nil && err != image.ErrFormat {
		xl.Infof("parse image failed. %v", err)
		return nil, err
	}

	var req FaceSearchV2Req
	req.Data.URI = args.Data.URI
	if img.URI != nil {
		req.Data.URI = *img.URI
	}
	req.Params.Limit = args.Params.Limit
	if req.Params.Limit <= 0 {
		req.Params.Limit = 1
	}

	ret, err := s.postSearchPolitician(ctx, req, env.Uid, env.Utype)

	if err == nil && env.UserInfo.Utype != utility.NoChargeUtype {
		if ret.Result.Review == true {
			server.SetStateHeader(env.W.Header(), "SEARCH_POLITICIAN-DEPEND", 1)
		} else {
			server.SetStateHeader(env.W.Header(), "SEARCH_POLITICIAN-CERTAIN", 1)
		}
	}
	xl.Infof("ret:%#v", ret)
	return &ret, err
}
