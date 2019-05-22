package censor

import (
	"context"
	"sync"

	"github.com/go-kit/kit/endpoint"

	xlog "github.com/qiniu/xlog.v1"

	"qiniu.com/argus/com/util"
	errors "qiniu.com/argus/service/service"
	"qiniu.com/argus/service/service/image/ads"
	"qiniu.com/argus/service/service/image/politician"
	"qiniu.com/argus/service/service/image/pulp"
	"qiniu.com/argus/service/service/image/terror"

	pimage "qiniu.com/argus/service/service/image"
)

type ImageCensorReq struct {
	Data struct {
		IMG pimage.Image
		// URI string `json:"uri"`
	} `json:"data"`
	Params struct {
		Type   []string `json:"type,omitempty"`
		Detail bool     `json:"detail"`
	} `json:"params,omitempty"`
}

type ImageCensorDetailResp struct {
	Type   string      `json:"type"`
	Label  int         `json:"label"`
	Class  string      `json:"class,omitempty"`
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

type CensorService interface {
	Censor(ctx context.Context, img ImageCensorReq) (ImageCensorResp, error)
	PremierCensor(ctx context.Context, img IPremierCensorRequest) (CensorResponse, error)
}

var _ CensorService = CensorEndpoints{}

type CensorEndpoints struct {
	CensorEP        endpoint.Endpoint
	PremierCensorEP endpoint.Endpoint
}

func (ends CensorEndpoints) Censor(ctx context.Context, img ImageCensorReq) (ImageCensorResp, error) {
	response, err := ends.CensorEP(ctx, img)
	if err != nil {
		return ImageCensorResp{}, err
	}
	resp := response.(ImageCensorResp)
	return resp, nil
}

func (ends CensorEndpoints) PremierCensor(ctx context.Context, img IPremierCensorRequest) (CensorResponse, error) {
	response, err := ends.PremierCensorEP(ctx, img)
	if err != nil {
		return CensorResponse{}, err
	}
	resp := response.(CensorResponse)
	return resp, nil
}

var _ CensorService = censorService{}

var (
	DEFAULT = Config{
		AdsConfig:        ads.DEFAULT,
		PulpConfig:       pulp.DEFAULT,
		PoliticianConfig: politician.DEFAULT,
		TerrorConfig:     terror.DEFAULT,
	}
)

type Config struct {
	PulpConfig       pulp.Config       `json:"pulp_config"`
	PoliticianConfig politician.Config `json:"politician_config"`
	TerrorConfig     terror.Config     `json:"terror_config"`
	AdsConfig        ads.Config        `json:"ads_config"`
}

type censorService struct {
	loadedService []string
	scenes        map[string]struct{}
	pulp.PulpService
	politician.FaceSearchService
	terror.TerrorService
	ads.AdsService
}

func NewCensorService(
	conf Config,
	eps pulp.EvalPulpService,
	epfs pulp.EvalPulpFilterService,
	efds politician.EvalFaceDetectService,
	effs politician.EvalFaceFeatureService,
	epts politician.EvalPoliticianService,
	etms terror.EvalTerrorMixupService,
	etds terror.EvalTerrorDetectService,
	eaqs ads.EvalAdsQrcodeService,
	eads ads.EvalAdsDetectService,
	ears ads.EvalAdsRecognitionService,
	eacs ads.EvalAdsClassifierService,
) (cs CensorService, err error) {

	var (
		pps           pulp.PulpService
		pts           politician.FaceSearchService
		ts            terror.TerrorService
		adss          ads.AdsService
		scenes        = make(map[string]struct{})
		loadedService = make([]string, 0)
	)

	if eps != nil && epfs != nil {
		scenes[PULP] = struct{}{}
		pps, err = pulp.NewPulpService(conf.PulpConfig, eps, epfs)
		if err != nil {
			return
		}
	}
	if efds != nil && effs != nil && epts != nil {
		scenes[POLITICIAN] = struct{}{}
		pts, err = politician.NewFaceSearchService(
			conf.PoliticianConfig,
			efds,
			effs,
			epts,
		)
		if err != nil {
			return
		}
	}

	if etms != nil && etds != nil {
		scenes[TERROR] = struct{}{}
		ts, err = terror.NewTerrorService(
			conf.TerrorConfig,
			etms,
			etds,
		)
		if err != nil {
			return
		}

	}

	if eaqs != nil && eads != nil && ears != nil && eacs != nil {
		scenes[ADS] = struct{}{}
		adss, err = ads.NewAdsService(
			conf.AdsConfig,
			eaqs,
			eads,
			ears,
			eacs,
		)
		if err != nil {
			return
		}
	}

	for s, _ := range scenes {
		loadedService = append(loadedService, s)
	}

	return censorService{
		loadedService:     loadedService,
		scenes:            scenes,
		PulpService:       pps,
		FaceSearchService: pts,
		TerrorService:     ts,
		AdsService:        adss,
	}, nil
}

func (s censorService) Censor(ctx context.Context, args ImageCensorReq) (ret ImageCensorResp, err error) {

	var xl = xlog.FromContextSafe(ctx)

	xl.Infof("censor args:%v", args)

	const (
		PULP       string = "pulp"
		TERROR     string = "terror"
		POLITICIAN string = "politician"
	)

	ret = ImageCensorResp{}
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
		args.Params.Type = s.loadedService
	}

	for _, typ := range args.Params.Type {
		switch typ {
		case PULP:
			if s.PulpService == nil {
				err = errors.ErrUriNotSupported("Request not supported: No such service")
				return
			}
			wg.Add(1)
			go func(ctx context.Context) {
				defer wg.Done()

				var (
					req pulp.PulpReq
				)
				req.Data.IMG = args.Data.IMG
				// if img.URI != nil {
				// 	req.Data.URI = *img.URI
				// }
				resp, err1 := s.Pulp(ctx, req)

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

			}(util.SpawnContext(ctx))
		case TERROR:
			if s.TerrorService == nil {
				err = errors.ErrUriNotSupported("Request not supported: No such service")
				return
			}
			wg.Add(1)
			go func(ctx context.Context) {
				defer wg.Done()

				var (
					req terror.TerrorReq
				)
				req.Data.IMG = args.Data.IMG
				req.Params.Detail = args.Params.Detail
				// if img.URI != nil {
				// 	req.Data.URI = *img.URI
				// }
				resp, err1 := s.Terror(ctx, req)

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

			}(util.SpawnContext(ctx))
		case POLITICIAN:
			if s.FaceSearchService == nil {
				err = errors.ErrUriNotSupported("Request not supported: No such service")
				return
			}
			wg.Add(1)
			go func(ctx context.Context) {
				defer wg.Done()
				var req politician.Req
				req.Data.IMG = args.Data.IMG
				// if img.URI != nil {
				// 	req.Data.URI = *img.URI
				// }
				resp, err1 := s.FaceSearch(ctx, req)

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
			}(util.SpawnContext(ctx))
		}
	}

	wg.Wait()
	return
}
