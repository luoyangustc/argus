package censor

import (
	"context"
	"image"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/qiniu/xlog.v1"
	"qiniu.com/auth/authstub.v1"

	"qiniu.com/argus/argus/com/util"
	"qiniu.com/argus/utility/evals"
	"qiniu.com/argus/utility/server"
)

type TerrorComplexResp struct {
	Code    int                 `json:"code"`
	Message string              `json:"message"`
	Result  TerrorComplexResult `json:"result"`
}

type TerrorComplexResult struct {
	Label   int `json:"label"`
	Classes []struct {
		Class string  `json:"class,omitempty"`
		Score float32 `json:"score,omitempty"`
	} `json:"classes,omitempty"`
	Score  float32 `json:"score"`
	Review bool    `json:"review"`
}

func (t *TerrorComplexResp) _Final(threshold float32) {
	if t.Result.Score < threshold {
		t.Result.Review = true
	}
}

func (s *Service) postTerrorComplex(
	ctx context.Context, req evals.SimpleReq, uid, utype uint32, detail bool,
) (ret TerrorComplexResp, err error) {

	var (
		xl               = xlog.FromContextSafe(ctx)
		wg               sync.WaitGroup
		derr, cerr       error
		cReview, cNormal bool
		lock             sync.Mutex
		cScore, dScore   float32
	)

	wg.Add(1)
	go func(ctx context.Context) {
		defer wg.Done()
		_t1 := time.Now()
		pdResp, pderr := s.eTerrorPreDet.Eval(ctx, req, uid, utype)
		server.ResponseTimeAtClient("terror", "terror_detect", "").
			Observe(server.DurationAsFloat64(time.Since(_t1)))
		if pderr != nil {
			xl.Errorf("query terror predetect error:%v", pderr)
		}

		var (
			dResp   evals.TerrorDetectResp
			detects = make(map[string]float32)
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
			server.HttpRequestsCounter("terror", "terror-detect", server.FormatError(derr)).Inc()
			for _, d := range dResp.Result.Detections {
				if score, ok := detects[d.Class]; ok {
					if d.Score > score {
						detects[d.Class] = d.Score
					}
				} else {
					detects[d.Class] = d.Score
				}
			}
			{
				lock.Lock()
				for class, score := range detects {
					if score > s.Config.TerrorThreshold {
						if detail {
							ret.Result.Classes = append(ret.Result.Classes, struct {
								Class string  `json:"class,omitempty"`
								Score float32 `json:"score,omitempty"`
							}{
								Class: class,
								Score: score,
							})
						}
						if score > dScore {
							dScore = score
						}
					}
				}
				lock.Unlock()
				return
			}
		}

		xl.Infof("call /v1/eval/terror-detect no target with rate larger than %v detected :%v",
			s.Config.TerrorThreshold, dResp)
	}(util.SpawnContext(ctx))
	wg.Add(1)
	go func(ctx context.Context) {
		defer wg.Done()
		var (
			cResp evals.TerrorClassifyResp
		)
		_t3 := time.Now()
		cResp, cerr := s.eTerrorClassify.Eval(ctx, req, uid, utype)
		server.ResponseTimeAtClient("terror", "terror_classify", "").
			Observe(server.DurationAsFloat64(time.Since(_t3)))
		server.HttpRequestsCounter("terror", "terror-classify", server.FormatError(cerr)).Inc()
		if cerr == nil && (cResp.Code == 0 || cResp.Code/100 == 2) && len(cResp.Result.Confidences) > 0 {
			cRespClass := strings.TrimSpace(cResp.Result.Confidences[0].Class)
			cScore = cResp.Result.Confidences[0].Score
			if cRespClass == "normal" {
				cNormal = true
			} else {
				if detail {
					lock.Lock()
					ret.Result.Classes = append(ret.Result.Classes, struct {
						Class string  `json:"class,omitempty"`
						Score float32 `json:"score,omitempty"`
					}{
						Class: cResp.Result.Confidences[0].Class,
						Score: cResp.Result.Confidences[0].Score,
					})
					lock.Unlock()
				}
			}
			if cResp.Result.Confidences[0].Index == -1 {
				cReview = true
			}
		}
	}(util.SpawnContext(ctx))
	wg.Wait()
	if cerr != nil && derr != nil { //直接返回服务端的错误内容
		xl.Errorf(
			"both of classify and detect service failed,classify error:%v, detect error:%v",
			cerr, derr)
		err = cerr
		return
	}
	if dScore > 0 || !cNormal {
		ret.Result.Label = 1
		ret.Result.Score = dScore
		if !cNormal && cScore > dScore {
			ret.Result.Score = cScore
		}
	} else {
		ret.Result.Score = cScore
	}
	ret._Final(s.Config.TerrorThreshold)
	ret.Result.Review = ret.Result.Review && cReview
	sort.Slice(ret.Result.Classes, func(i, j int) bool { return ret.Result.Classes[i].Score > ret.Result.Classes[j].Score })

	return
}

func (s *Service) PostTerrorComplex(
	ctx context.Context, args *TerrorReq, env *authstub.Env,
) (*TerrorComplexResp, error) {

	var err error
	defer func(begin time.Time) {
		server.ResponseTimeAtServer("terror-complex", "").
			Observe(server.DurationAsFloat64(time.Since(begin)))
		server.HttpRequestsCounter("terror-complex", "", server.FormatError(err)).Inc()
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
	ret, err := s.postTerrorComplex(ctx, req, env.Uid, env.Utype, args.Params.Detail)

	if err == nil && env.UserInfo.Utype != server.NoChargeUtype {
		if ret.Result.Review {
			server.SetStateHeader(env.W.Header(), "TERROR_COMPLEX-Depend", 1)
		} else {
			server.SetStateHeader(env.W.Header(), "TERROR_COMPLEX-Certain", 1)
		}
	}

	return &ret, err
}
