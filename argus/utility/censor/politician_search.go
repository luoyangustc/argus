package censor

import (
	"context"
	"encoding/base64"
	"image"
	"strings"
	"sync"

	"github.com/qiniu/xlog.v1"
	"qiniu.com/auth/authstub.v1"

	"qiniu.com/argus/argus/com/util"
	"qiniu.com/argus/utility/evals"
	"qiniu.com/argus/utility/server"
)

// FaceSearchReq ...
type FaceSearchReq struct {
	Data struct {
		URI string `json:"uri"`
	} `json:"data"`
}

// FaceSearchResp ...
type FaceSearchResp struct {
	Code    int              `json:"code"`
	Message string           `json:"message"`
	Result  FaceSearchResult `json:"result"`
}

// FaceSearchResult ...
type FaceSearchResult struct {
	Review     bool               `json:"review"`
	Detections []FaceSearchDetail `json:"detections"`
}

// FaceSearchDetail ...
type FaceSearchDetail struct {
	BoundingBox struct {
		Pts   [][2]int `json:"pts"`
		Score float32  `json:"score"`
	} `json:"boundingBox"`
	Value struct {
		Name   string  `json:"name,omitempty"`
		Group  string  `json:"group,omitempty"`
		Score  float32 `json:"score"`
		Review bool    `json:"review"`
	} `json:"value"`
	Sample *FaceSearchDetailSample `json:"sample,omitempty"`
}

type FaceSearchDetailSample struct {
	URL string   `json:"url"`
	Pts [][2]int `json:"pts"`
}

func (s *Service) postFaceSearchPolitician(
	ctx context.Context, req evals.SimpleReq, uid, utype uint32,
) (ret FaceSearchResp, err error) {

	var (
		xl = xlog.FromContextSafe(ctx)
	)

	dResp, err := s.eFaceDet.Eval(ctx, req, uid, utype)
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
	ret.Result.Detections = make([]FaceSearchDetail, 0, len(dResp.Result.Detections))
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

			var mReq evals.SimpleReq
			mReq.Data.URI = "data:application/octet-stream;base64," +
				base64.StdEncoding.EncodeToString(ff)
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

			var (
				name  string
				score float32
				url   string
				group string
				pts   [][2]int
			)
			if mfResp, ok := mResp.(evals.FaceSearchRespV2); ok {
				if len(mfResp.Result.Confidences) == 0 || mfResp.Result.Confidences[0].Class == "" {
					xl.Errorf("unexpected s.politician.Eval result, len(mResp.Result.Confidences):%v", len(mfResp.Result.Confidences))
					return
				}
				score = mfResp.Result.Confidences[0].Score
				url = mfResp.Result.Confidences[0].Sample.URL
				pts = mfResp.Result.Confidences[0].Sample.Pts
				name = mfResp.Result.Confidences[0].Class
				group = mfResp.Result.Confidences[0].Group
			} else if mfResp, ok := mResp.(evals.FaceSearchResp); ok {
				if mfResp.Result.Class == "" {
					xl.Errorf("unexpected s.politician.Eval result, len(mResp.Result.Confidences):%v", mfResp)
					return
				}
				score = mfResp.Result.Score
				url = mfResp.Result.Sample.URL
				pts = mfResp.Result.Sample.Pts
				name = mfResp.Result.Class
				group = mfResp.Result.Group
			} else {
				return
			}

			detail := FaceSearchDetail{
				BoundingBox: struct {
					Pts   [][2]int `json:"pts"`
					Score float32  `json:"score"`
				}{
					Pts:   face.Pts,
					Score: face.Score,
				},
			}

			if score >= s.Config.PoliticianThreshold[1] {
				detail.Value.Name = name
				detail.Value.Group = group
				detail.Sample = &FaceSearchDetailSample{
					URL: url,
					Pts: pts,
				}
			}
			detail.Value.Score = score
			if score > s.Config.PoliticianThreshold[0] &&
				score < s.Config.PoliticianThreshold[2] {
				detail.Value.Review = true
				ret.Result.Review = true
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

func (s *Service) PostFaceSearchPolitician(
	ctx context.Context, args *FaceSearchReq, env *authstub.Env,
) (*FaceSearchResp, error) {

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
	ret, err := s.postFaceSearchPolitician(ctx, req, env.Uid, env.Utype)

	if err == nil && env.UserInfo.Utype != server.NoChargeUtype {
		if ret.Result.Review == true {
			server.SetStateHeader(env.W.Header(), "FACE_SEARCH_POLITICIAN-DEPEND", 1)
		} else {
			server.SetStateHeader(env.W.Header(), "FACE_SEARCH_POLITICIAN-CERTAIN", 1)
		}
	}
	return &ret, err
}
