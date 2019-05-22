package censor

import (
	"context"
	"image"
	"strings"
	"sync"
	"time"

	"qiniu.com/argus/argus/com/util"
	"qiniu.com/argus/utility/evals"
	"qiniu.com/argus/utility/server"
	"qiniu.com/auth/authstub.v1"
)

type ImageCensorReq struct {
	Data struct {
		URI string `json:"uri"`
	} `json:"data"`
	Params struct {
		Type   []string `json:"type,omitempty"`
		Detail bool     `json:"detail"`
	} `json:"params,omitempty"`
}

type ImageCensorDetailResp struct {
	Type    string `json:"type"`
	Label   int    `json:"label"`
	Class   string `json:"class,omitempty"`
	Classes []struct {
		Class string  `json:"class,omitempty"`
		Score float32 `json:"score,omitempty"`
	} `json:"classes,omitempty"`
	Score  float32     `json:"score"`
	Review bool        `json:"review"`
	More   interface{} `json:"more,omitempty"`
}

type ImageCensorResp struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Result  struct {
		Label   int                     `json:"label"`
		Score   float32                 `json:"score"`
		Review  bool                    `json:"review"`
		Details []ImageCensorDetailResp `json:"details"`
	} `json:"result"`
}

func (s *Service) PostImageCensor(
	ctx context.Context, args *ImageCensorReq, env *authstub.Env,
) (ret *ImageCensorResp, err error) {

	defer func(begin time.Time) {
		server.ResponseTimeAtServer("image_censor", "").
			Observe(server.DurationAsFloat64(time.Since(begin)))
		server.HttpRequestsCounter("image_censor", "", server.FormatError(err)).Inc()
	}(time.Now())

	ctx, xl := util.CtxAndLog(ctx, env.W, env.Req)
	xl.Infof("censor args:%v", args)
	if strings.TrimSpace(args.Data.URI) == "" {
		err = server.ErrArgs
		return nil, err
	}

	const (
		PULP           string = "pulp"
		TERROR         string = "terror"
		TERROR_COMPLEX string = "terror-complex"
		POLITICIAN     string = "politician"
	)

	ret = &ImageCensorResp{}
	ret.Result.Details = make([]ImageCensorDetailResp, 0)
	var (
		wg   sync.WaitGroup
		lock = new(sync.Mutex)

		update = func(label int, score float32, review bool) {
			if ret.Result.Label == 0 {
				if label == 1 {
					ret.Result.Label = 1
					ret.Result.Score = score
					ret.Result.Review = review
				} else {
					if ret.Result.Score < score {
						ret.Result.Score = score
						ret.Result.Review = review
					}
				}
			} else {
				if label == 1 && ret.Result.Score < score {
					ret.Result.Score = score
					ret.Result.Review = review
				}
			}
		}
	)

	if args.Params.Type == nil || len(args.Params.Type) == 0 {
		args.Params.Type = []string{PULP, TERROR, POLITICIAN}
	}

	uri, _ := server.ImproveURI(args.Data.URI, env.Uid)
	img, err := s.ParseImage(ctx, uri)
	if err != nil && err != image.ErrFormat {
		xl.Infof("parse image failed. %v", err)
		return nil, err
	}

	for _, typ := range args.Params.Type {
		switch typ {
		case PULP:
			wg.Add(1)
			go func(ctx context.Context) {
				defer wg.Done()

				var (
					req evals.PulpReq
				)
				req.Data.URI = args.Data.URI
				if img.URI != nil {
					req.Data.URI = *img.URI
				}
				resp, err1 := s.postPulp(ctx, &req, img, env.Uid, env.Utype)

				lock.Lock()
				defer lock.Unlock()

				if err1 != nil {
					err = err1
					return
				}
				ret.Result.Details = append(ret.Result.Details,
					ImageCensorDetailResp{
						Type:   PULP,
						Label:  resp.Result.Label,
						Score:  resp.Result.Score,
						Review: resp.Result.Review,
					})
				var label = 0
				if resp.Result.Label == 0 {
					label = 1
				}
				update(label, resp.Result.Score, resp.Result.Review)

				if env.UserInfo.Utype != server.NoChargeUtype {
					if resp.Result.Review {
						server.SetStateHeader(env.W.Header(), "PULP-Depend", 1)
					} else {
						server.SetStateHeader(env.W.Header(), "PULP-Certain", 1)
					}
				}

			}(util.SpawnContext(ctx))
		case TERROR:
			wg.Add(1)
			go func(ctx context.Context) {
				defer wg.Done()

				var (
					req evals.SimpleReq
				)
				req.Data.URI = args.Data.URI
				if img.URI != nil {
					req.Data.URI = *img.URI
				}
				resp, err1 := s.postTerror(ctx, req, env.Uid, env.Utype, args.Params.Detail)

				lock.Lock()
				defer lock.Unlock()

				if err1 != nil {
					err = err1
					return
				}
				ret.Result.Details = append(ret.Result.Details,
					ImageCensorDetailResp{
						Type:   TERROR,
						Label:  resp.Result.Label,
						Class:  resp.Result.Class,
						Score:  resp.Result.Score,
						Review: resp.Result.Review,
					})
				update(resp.Result.Label, resp.Result.Score, resp.Result.Review)

				if env.UserInfo.Utype != server.NoChargeUtype {
					if resp.Result.Review {
						server.SetStateHeader(env.W.Header(), "TERROR-Depend", 1)
					} else {
						server.SetStateHeader(env.W.Header(), "TERROR-Certain", 1)
					}
				}

			}(util.SpawnContext(ctx))
		case TERROR_COMPLEX:
			wg.Add(1)
			go func(ctx context.Context) {
				defer wg.Done()

				var (
					req evals.SimpleReq
				)
				req.Data.URI = args.Data.URI
				if img.URI != nil {
					req.Data.URI = *img.URI
				}
				resp, err1 := s.postTerrorComplex(ctx, req, env.Uid, env.Utype, args.Params.Detail)

				lock.Lock()
				defer lock.Unlock()

				if err1 != nil {
					err = err1
					return
				}
				ret.Result.Details = append(ret.Result.Details,
					ImageCensorDetailResp{
						Type:    TERROR_COMPLEX,
						Label:   resp.Result.Label,
						Classes: resp.Result.Classes,
						Score:   resp.Result.Score,
						Review:  resp.Result.Review,
					})
				update(resp.Result.Label, resp.Result.Score, resp.Result.Review)

				if env.UserInfo.Utype != server.NoChargeUtype {
					if resp.Result.Review {
						server.SetStateHeader(env.W.Header(), "TERROR_COMPLEX-Depend", 1)
					} else {
						server.SetStateHeader(env.W.Header(), "TERROR_COMPLEX-Certain", 1)
					}
				}

			}(util.SpawnContext(ctx))
		case POLITICIAN:
			wg.Add(1)
			go func(ctx context.Context) {
				defer wg.Done()
				var req evals.SimpleReq
				req.Data.URI = args.Data.URI
				if img.URI != nil {
					req.Data.URI = *img.URI
				}
				resp, err1 := s.postFaceSearchPolitician(ctx, req, env.Uid, env.Utype)

				lock.Lock()
				defer lock.Unlock()

				if err1 != nil {
					err = err1
					return
				}
				var (
					label int
					score float32
				)

				for _, d := range resp.Result.Detections {
					if len(d.Value.Name) > 0 {
						label = 1
					}
					if score < d.Value.Score {
						score = d.Value.Score
					}
				}
				if !resp.Result.Review && label == 0 && score != 0 {
					score = 1 - score
				}
				if score == 0 {
					score = 0.99
				}

				ret.Result.Details = append(ret.Result.Details,
					ImageCensorDetailResp{
						Type:   POLITICIAN,
						Label:  label,
						Score:  score,
						Review: resp.Result.Review,
						More:   resp.Result.Detections,
					})
				update(label, score, resp.Result.Review)

				if env.UserInfo.Utype != server.NoChargeUtype {
					if resp.Result.Review == true {
						server.SetStateHeader(env.W.Header(), "FACE_SEARCH_POLITICIAN-DEPEND", 1)
					} else {
						server.SetStateHeader(env.W.Header(), "FACE_SEARCH_POLITICIAN-CERTAIN", 1)
					}
				}

			}(util.SpawnContext(ctx))
		}
	}

	wg.Wait()
	return
}
