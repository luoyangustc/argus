package pulp

import (
	"context"
	"sort"

	xlog "github.com/qiniu/xlog.v1"
	. "qiniu.com/argus/service/service"
	pimage "qiniu.com/argus/service/service/image"
	"qiniu.com/argus/utility/evals"
)

const (
	TAG_NORMAL = "normal" //正常
	TAG_PULP   = "pulp"   // 色情
	TAG_SEXY   = "sexy"   // 性感
)

//----------------------------------------------------------------------------//

type PulpThreshold struct {
	Pulp   *float32 `json:"pulp,omitempty"`
	Sexy   *float32 `json:"sexy,omitempty"`
	Normal *float32 `json:"normal,omitempty"`
}

type PulpCensorResult struct {
	Label string  `json:"label"`
	Score float32 `json:"score"`
}

func (s pulpService) PulpCensor(ctx context.Context, req pimage.ImageCensorReq) (resp pimage.SceneResult, err error) {

	var req1 PulpReq
	req1.Data.IMG = req.Data.IMG
	var (
		sugcfg = &s.Config.SugConfig
		xl     = xlog.FromContextSafe(ctx)
	)

	resp1, err := s.EvalPulpFilter(ctx, req1)
	if err != nil {
		err = ErrInternal(err.Error())
		xl.Errorf("call pulp filter failed. %v", err)
		return
	}

	if resp1.Result.Checkpoint == "endpoint" {
		return ConvPulp(ctx, sugcfg, resp1), nil
	}

	resp2, err := s.EvalPulp(ctx, req1)
	if err != nil {
		err = ErrInternal(err.Error())
		xl.Errorf("call pulp failed. %v", err)
		return
	}
	return ConvPulp(ctx, sugcfg, resp2), nil
}

func ConvPulp(ctx context.Context, sugcfg *pimage.SugConfig,
	resp evals.PulpResp) pimage.SceneResult {

	sceneResult := pimage.SceneResult{
		Suggestion: pimage.PASS,
	}

	sort.Slice(resp.Result.Confidences, func(i, j int) bool {
		return resp.Result.Confidences[i].Score > resp.Result.Confidences[j].Score
	})

	item := resp.Result.Confidences[0]
	if item.Class == "" {
		return sceneResult
	}

	detail := &pimage.Detail{
		Suggestion: pimage.PASS,
		Label:      item.Class,
		Score:      item.Score,
	}

	ok := detail.SetSuggestion(sugcfg)
	if !ok {
		return sceneResult
	}

	sceneResult.Details = append(sceneResult.Details, *detail)
	sceneResult.Suggestion = sceneResult.Suggestion.Update(detail.Suggestion)

	return sceneResult
}
