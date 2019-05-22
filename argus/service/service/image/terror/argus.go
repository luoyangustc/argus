package terror

import (
	"context"
	"sort"

	"github.com/go-kit/kit/endpoint"
	xlog "github.com/qiniu/xlog.v1"

	. "qiniu.com/argus/service/service"
	pimage "qiniu.com/argus/service/service/image"
	"qiniu.com/argus/utility/evals"
)

type TerrorReq struct {
	Data struct {
		IMG pimage.Image
		// URI string `json:"uri"`
	} `json:"data"`
	Params struct {
		Detail bool `json:"detail"`
	} `json:"params"`
}

type TerrorResp struct {
	Code    int          `json:"code"`
	Message string       `json:"message"`
	Result  TerrorResult `json:"result"`
}

type TerrorResult struct {
	Label  int     `json:"label"`
	Class  string  `json:"class,omitempty"`
	Score  float32 `json:"score"`
	Review bool    `json:"review"`
}

type TerrorService interface {
	Terror(ctx context.Context, args TerrorReq) (TerrorResp, error)
	TerrorCensor(ctx context.Context, args pimage.ImageCensorReq) (resp pimage.SceneResult, err error)
}

var _ TerrorService = TerrorEndpoints{}

type TerrorEndpoints struct {
	TerrorEP       endpoint.Endpoint
	TerrorCensorEP endpoint.Endpoint
}

func (ends TerrorEndpoints) Terror(ctx context.Context, args TerrorReq) (TerrorResp, error) {

	response, err := ends.TerrorEP(ctx, args)
	if err != nil {
		return TerrorResp{}, err
	}
	resp := response.(TerrorResp)
	return resp, nil
}

func (ends TerrorEndpoints) TerrorCensor(ctx context.Context, args pimage.ImageCensorReq) (resp pimage.SceneResult, err error) {
	response, err := ends.TerrorCensorEP(ctx, args)
	if err != nil {
		return pimage.SceneResult{}, err
	}
	resp = response.(pimage.SceneResult)
	return resp, nil
}

var _ TerrorService = terrorService{}

type Config struct {
	TerrorThreshold float32 `json:"terror_threshold"`
	pimage.SugConfig
}

var (
	RuleAlwaysPass = pimage.RuleConfig{
		SureThreshold:    0.85,
		AbandonThreshold: 0.3,
		SureSuggestion:   pimage.PASS,
		UnsureSuggestion: pimage.PASS,
	}
	RuleWithBlock = pimage.RuleConfig{
		SureThreshold:    0.85,
		AbandonThreshold: 0.3,
		SureSuggestion:   pimage.BLOCK,
		UnsureSuggestion: pimage.REVIEW,
	}
	RuleAlwaysReview = pimage.RuleConfig{
		SureThreshold:    0.85,
		AbandonThreshold: 0.3,
		SureSuggestion:   pimage.REVIEW,
		UnsureSuggestion: pimage.REVIEW,
	}
	DEFAULT = Config{
		TerrorThreshold: 0.85,
		SugConfig: pimage.SugConfig{
			CensorBy: "label",
			Rules: map[string]pimage.RuleConfig{
				"normal":             RuleAlwaysPass,
				"bomb":               RuleAlwaysPass,
				"army":               RuleAlwaysPass,
				"bloodiness_animal":  RuleAlwaysPass,
				"fire_weapon":        RuleAlwaysPass,
				"guns":               RuleWithBlock,
				"self_burning":       RuleWithBlock,
				"beheaded":           RuleWithBlock,
				"illegal_flag":       RuleWithBlock,
				"fight_person":       RuleAlwaysReview,
				"fight_police":       RuleAlwaysReview,
				"anime_knives":       RuleAlwaysReview,
				"knives":             RuleAlwaysReview,
				"anime_guns":         RuleAlwaysReview,
				"bloodiness":         RuleAlwaysReview,
				"anime_bloodiness":   RuleAlwaysReview,
				"special_clothing":   RuleAlwaysReview,
				"march_crowed":       RuleAlwaysReview,
				"special_characters": RuleAlwaysReview,
			},
		},
	}
)

type terrorService struct {
	Config
	EvalTerrorMixupService
	EvalTerrorDetectService
}

func NewTerrorService(
	conf Config,
	s1 EvalTerrorMixupService,
	s2 EvalTerrorDetectService,
) (TerrorService, error) {
	return terrorService{
		Config:                  conf,
		EvalTerrorMixupService:  s1,
		EvalTerrorDetectService: s2,
	}, nil
}

func (t *TerrorResp) _Final(threshold float32) {
	if t.Result.Score < threshold {
		t.Result.Review = true
	}
}

func (s terrorService) Terror(ctx context.Context, args TerrorReq) (ret TerrorResp, err error) {

	var (
		req    SimpleReq
		detail bool
	)
	detail = args.Params.Detail
	req.Data.IMG = args.Data.IMG

	////////////////////////////////////////////////////////////////
	var (
		xl    = xlog.FromContextSafe(ctx)
		mResp TerrorMixupResp
	)

	mResp, merr := s.EvalTerrorMixup(ctx, req)

	if merr != nil {
		err = ErrInternal(merr.Error())
		xl.Errorf("call terror_mixup failed. %v", err)
		return
	}

	if len(mResp.Result.Confidences) == 0 {
		xl.Errorf("unexpected terror_mixup resp:%v", mResp)
		err = ErrInternal("classify failed")
		return
	}

	sort.Slice(mResp.Result.Confidences, func(i, j int) bool {
		return mResp.Result.Confidences[i].Score > mResp.Result.Confidences[j].Score
	})
	ret.Result.Score = mResp.Result.Confidences[0].Score
	if mResp.Result.Confidences[0].Class != "normal" {
		ret.Result.Label = 1
	}
	ret._Final(s.TerrorThreshold)
	if detail {
		ret.Result.Class = mResp.Result.Confidences[0].Class
	}
	if mResp.Result.Checkpoint == "endpoint" {
		return
	}
	var (
		dResp evals.TerrorDetectResp
		derr  error
	)

	dResp, derr = s.EvalTerrorDetect(ctx, req)
	if derr != nil {
		xl.Errorf("call /v1/eval/terror-detect error resp : %v", derr)
		err = ErrInternal(derr.Error())
		return
	}
	if len(dResp.Result.Detections) != 0 {
		sort.Slice(dResp.Result.Detections, func(i, j int) bool {
			return dResp.Result.Detections[i].Score > dResp.Result.Detections[j].Score
		})
		ret.Result.Label = 1
		ret.Result.Review = false
		ret.Result.Score = dResp.Result.Detections[0].Score
		ret._Final(s.TerrorThreshold)
		if detail {
			ret.Result.Class = dResp.Result.Detections[0].Class
		}
	}
	ret.Message = "success"

	return

}
