package terror_complex

import (
	"context"
	"sort"
	"strings"
	"sync"

	"github.com/go-kit/kit/endpoint"
	xlog "github.com/qiniu/xlog.v1"

	"qiniu.com/argus/com/util"
	. "qiniu.com/argus/service/service"
	"qiniu.com/argus/service/service/image/terror"
	"qiniu.com/argus/utility/evals"
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

type TerrorComplexService interface {
	TerrorComplex(ctx context.Context, args terror.TerrorReq) (TerrorComplexResp, error)
}

var _ TerrorComplexService = TerrorComplexEndpoints{}

type TerrorComplexEndpoints struct {
	TerrorComplexEP endpoint.Endpoint
}

func (ends TerrorComplexEndpoints) TerrorComplex(ctx context.Context, args terror.TerrorReq) (TerrorComplexResp, error) {

	response, err := ends.TerrorComplexEP(ctx, args)
	if err != nil {
		return TerrorComplexResp{}, err
	}
	resp := response.(TerrorComplexResp)
	return resp, nil
}

var _ TerrorComplexService = terrorComplexService{}

type Config struct {
	TerrorThreshold float32 `json:"terror_threshold"`
}

var (
	DEFAULT = Config{
		TerrorThreshold: 0.83,
	}
)

type terrorComplexService struct {
	Config
	terror.EvalTerrorDetectService
	terror.EvalTerrorMixupService
}

func NewTerrorComplexService(
	conf Config,
	s1 terror.EvalTerrorDetectService,
	s2 terror.EvalTerrorMixupService,
) (TerrorComplexService, error) {
	return terrorComplexService{
		Config:                  conf,
		EvalTerrorDetectService: s1,
		EvalTerrorMixupService:  s2,
	}, nil
}

func (t *TerrorComplexResp) _Final(threshold float32) {
	if t.Result.Score < threshold {
		t.Result.Review = true
	}
}

func (s terrorComplexService) TerrorComplex(ctx context.Context, args terror.TerrorReq) (ret TerrorComplexResp, err error) {

	var (
		req    terror.SimpleReq
		detail bool
	)
	detail = args.Params.Detail
	req.Data.IMG = args.Data.IMG

	var (
		xl             = xlog.FromContextSafe(ctx)
		wg             sync.WaitGroup
		derr, cerr     error
		cNormal        bool
		lock           sync.Mutex
		cScore, dScore float32
	)

	wg.Add(1)
	go func(ctx context.Context) {
		defer wg.Done()
		var (
			dResp   evals.TerrorDetectResp
			detects = make(map[string]float32)
		)
		dResp, derr = s.EvalTerrorDetect(ctx, req)
		if derr != nil {
			xl.Errorf("call /v1/eval/terror-detect error resp : %v", derr)
			err = ErrInternal(derr.Error())
			return
		}
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
			xl.Infof("call /v1/eval/terror-detect %v detected :%v",
				s.Config.TerrorThreshold, dResp)
			return
		}

	}(util.SpawnContext(ctx))
	wg.Add(1)
	go func(ctx context.Context) {
		defer wg.Done()
		var (
			cResp terror.TerrorMixupResp
		)
		cResp, cerr := s.EvalTerrorMixup(ctx, req)
		if cerr == nil && (cResp.Code == 0 || cResp.Code/100 == 2) && len(cResp.Result.Confidences) > 0 {
			sort.Slice(cResp.Result.Confidences, func(i, j int) bool {
				return cResp.Result.Confidences[i].Score > cResp.Result.Confidences[j].Score
			})
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
		}
	}(util.SpawnContext(ctx))
	wg.Wait()
	if cerr != nil && derr != nil {
		xl.Errorf(
			"both of classify and detect service failed,classify error:%v, detect error:%v",
			cerr, derr)
		err = ErrInternal(cerr.Error())
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
	sort.Slice(ret.Result.Classes, func(i, j int) bool { return ret.Result.Classes[i].Score > ret.Result.Classes[j].Score })
	return
}
