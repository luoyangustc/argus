package pulp

import (
	"context"

	"github.com/go-kit/kit/endpoint"
	xlog "github.com/qiniu/xlog.v1"

	. "qiniu.com/argus/service/service"
	pimage "qiniu.com/argus/service/service/image"
	"qiniu.com/argus/utility/evals"
)

type PulpReq struct {
	Data struct {
		IMG pimage.Image
		// URI string `json:"uri"`
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

type PulpService interface {
	Pulp(ctx context.Context, req PulpReq) (PulpResp, error)
	PulpCensor(ctx context.Context, req pimage.ImageCensorReq) (pimage.SceneResult, error)
}

var _ PulpService = PulpEndpoints{}

type PulpEndpoints struct {
	PulpEP       endpoint.Endpoint
	PulpCensorEP endpoint.Endpoint
}

func (ends PulpEndpoints) Pulp(ctx context.Context, req PulpReq) (PulpResp, error) {
	response, err := ends.PulpEP(ctx, req)
	if err != nil {
		return PulpResp{}, err
	}
	resp := response.(PulpResp)
	return resp, nil
}
func (ends PulpEndpoints) PulpCensor(ctx context.Context, req pimage.ImageCensorReq) (resp pimage.SceneResult, err error) {
	response, err := ends.PulpCensorEP(ctx, req)
	if err != nil {
		return pimage.SceneResult{}, err
	}
	resp = response.(pimage.SceneResult)
	return resp, nil
}

var _ PulpService = pulpService{}

type Config struct {
	PulpReviewThreshold float32 `json:"pulp_review_threshold"`
	pimage.SugConfig
}

var (
	RuleAlwaysPass = pimage.RuleConfig{
		SureThreshold:    0.6,
		AbandonThreshold: 0.22,
		SureSuggestion:   pimage.PASS,
		UnsureSuggestion: pimage.PASS,
	}
	RuleWithBlock = pimage.RuleConfig{
		SureThreshold:    0.6,
		AbandonThreshold: 0.22,
		SureSuggestion:   pimage.BLOCK,
		UnsureSuggestion: pimage.REVIEW,
	}
	RuleAlwaysReview = pimage.RuleConfig{
		SureThreshold:    0.6,
		AbandonThreshold: 0.22,
		SureSuggestion:   pimage.REVIEW,
		UnsureSuggestion: pimage.REVIEW,
	}
	DEFAULT = Config{
		PulpReviewThreshold: 0.6,
		SugConfig: pimage.SugConfig{
			CensorBy: "label",
			Rules: map[string]pimage.RuleConfig{
				"normal": RuleAlwaysPass,
				"pulp":   RuleWithBlock,
				"sexy":   RuleAlwaysReview,
			},
		},
	}
)

type pulpService struct {
	Config
	EvalPulpService
	EvalPulpFilterService
}

func NewPulpService(
	conf Config,
	eps EvalPulpService,
	epfs EvalPulpFilterService,
) (PulpService, error) {
	return pulpService{Config: conf, EvalPulpService: eps, EvalPulpFilterService: epfs}, nil
}

func (s pulpService) Pulp(ctx context.Context, req PulpReq) (ret PulpResp, err error) {

	var (
		xl = xlog.FromContextSafe(ctx)
	)

	var (
		limit = req.Params.Limit
		eResp evals.PulpResp
		big   bool
	)
	req.Params.Limit = 3

	{
		eResp, err = s.EvalPulpFilter(ctx, req)
		if err != nil {
			xl.Errorf("call pulp filter error: %v,", err)
			return ret, ErrInternal(err.Error())
		}
		if eResp.Result.Checkpoint != "endpoint" {
			big = true //小模型没有返回确切结果
		}
	}

	if big {
		eResp, err = s.EvalPulp(ctx, req)
		if err != nil {
			xl.Errorf("call pulp error: %v,", err)
			return ret, ErrInternal(err.Error())
		}

		if eResp.Code != 0 && eResp.Code/100 != 2 {
			xl.Errorf("call pulp error: %v", eResp)
			return ret, ErrInternal(eResp.Message)
		}
	}

	if len(eResp.Result.Confidences) != 3 {
		xl.Errorf("call pulp error: %v", eResp)
		return ret, ErrInternal("unexpected EvalPulp result")
	}

	ret.Result.Label = eResp.Result.Confidences[0].Index
	ret.Result.Score = eResp.Result.Confidences[0].Score
	if eResp.Result.Confidences[0].Score < s.Config.PulpReviewThreshold {
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
