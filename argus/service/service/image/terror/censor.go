package terror

import (
	"context"
	"sort"

	xlog "github.com/qiniu/xlog.v1"

	. "qiniu.com/argus/service/service"
	pimage "qiniu.com/argus/service/service/image"
	"qiniu.com/argus/utility/evals"
)

//----------------------------------------------------------------------------//

func (s terrorService) TerrorCensor(ctx context.Context, args pimage.ImageCensorReq) (resp pimage.SceneResult, err error) {
	var req SimpleReq
	req.Data.IMG = args.Data.IMG

	var (
		sugcfg = &s.Config.SugConfig
		xl     = xlog.FromContextSafe(ctx)
		mResp  TerrorMixupResp
	)

	// 分类
	mResp, merr := s.EvalTerrorMixup(ctx, req)
	if merr != nil {
		err = ErrInternal(merr.Error())
		xl.Errorf("call terror_mixup failed. %v", err)
		return
	}

	if len(mResp.Result.Confidences) == 0 {
		xl.Errorf("unexpected terror_mixup resp:%v", mResp)
		err = ErrInternal(merr.Error())
		return
	}
	sr1 := ConvTerrorMixup(ctx, sugcfg, mResp)
	if mResp.Result.Checkpoint == "endpoint" {
		return *sr1, nil
	}
	// 继续检测
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
	sr2 := ConvTerrorDetect(ctx, sugcfg, dResp)
	sr3 := pimage.MergeDetails(ctx, sugcfg, sr1.Details, sr2.Details...)

	return *sr3, nil
}

func ConvTerrorMixup(ctx context.Context, sugcfg *pimage.SugConfig,
	resp TerrorMixupResp) *pimage.SceneResult {
	sceneResult := pimage.SceneResult{
		Suggestion: pimage.PASS,
	}

	sort.Slice(resp.Result.Confidences, func(i, j int) bool {
		return resp.Result.Confidences[i].Score > resp.Result.Confidences[j].Score
	})

	item := resp.Result.Confidences[0]
	if item.Class == "" {
		return &sceneResult
	}

	detail := &pimage.Detail{
		Suggestion: pimage.PASS,
		Label:      item.Class,
		Score:      item.Score,
	}

	ok := detail.SetSuggestion(sugcfg)
	if !ok {
		return &sceneResult
	}

	sceneResult.Details = append(sceneResult.Details, *detail)
	sceneResult.Suggestion = sceneResult.Suggestion.Update(detail.Suggestion)

	return &sceneResult
}

func ConvTerrorDetect(ctx context.Context, sugcfg *pimage.SugConfig,
	resp evals.TerrorDetectResp) *pimage.SceneResult {
	sceneResult := pimage.SceneResult{
		Suggestion: pimage.PASS,
	}
	for _, item := range resp.Result.Detections {
		if item.Class == "" {
			continue
		}

		detail := &pimage.Detail{
			Suggestion: pimage.PASS,
			Label:      item.Class,
			Score:      item.Score,
			Detections: []pimage.BoundingBox{
				pimage.BoundingBox{
					Pts:   item.Pts,
					Score: item.Score,
				},
			},
		}

		ok := detail.SetSuggestion(sugcfg)
		if !ok {
			continue
		}

		sceneResult.Details = append(sceneResult.Details, *detail)
		sceneResult.Suggestion = sceneResult.Suggestion.Update(detail.Suggestion)
	}
	return &sceneResult
}
