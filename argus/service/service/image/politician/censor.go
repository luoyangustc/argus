package politician

import (
	"context"
	"encoding/base64"
	"sync"

	xlog "github.com/qiniu/xlog.v1"
	"golang.org/x/sync/errgroup"
	"qiniu.com/argus/com/util"
	. "qiniu.com/argus/service/service"
	pimage "qiniu.com/argus/service/service/image"
	"qiniu.com/argus/utility/evals"
)

const (
	TAG_DOMESTIC_STATESMAN   = "domestic_statesman"   // 国内政治人物
	TAG_FOREIGN_STATESMAN    = "foreign_statesman"    // 国外政治人物
	TAG_AFFAIRS_OFFICIAL_GOV = "affairs_official_gov" //落马官员（政府）
	TAG_AFFAIRS_OFFICIAL_ENT = "affairs_official_ent" //落马官员（企事业）
	TAG_ANTI_CHINA_PEOPLE    = "anti_china_people"    //反华分子
	TAT_TERRORIST            = "terrorist"            //恐怖分子
	TAG_AFFAIRS_CELEBRITY    = "affairs_celebrity"    //劣迹艺人
	TAG_CHINESE_MARTYR       = "chinese_martyr"       //烈士
)

//----------------------------------------------------------------------------//
func (s faceSearchService) PoliticianCensor(ctx context.Context, req pimage.ImageCensorReq) (ret pimage.SceneResult, err error) {
	xl := xlog.FromContextSafe(ctx)

	var fdReq Req
	fdReq.Data.IMG = req.Data.IMG
	fdResp, err := s.EvalFaceDetect(ctx, fdReq)
	if err != nil {
		xl.Errorf("call facex-detect error: %v", err)
		return ret, ErrInternal(err.Error())
	}
	if fdResp.Code != 0 && fdResp.Code/100 != 2 {
		xl.Errorf("call facex-detect failed: %d %s", fdResp.Code, fdResp.Message)
		return ret, ErrInternal(fdResp.Message)
	}

	details := make([]pimage.Detail, 0)
	waiter := errgroup.Group{}
	mux := sync.Mutex{}

	for i := 0; i < len(fdResp.Result.Detections); i++ {
		ord := i
		waiter.Go(func() error {
			face := fdResp.Result.Detections[ord]
			ctx2 := util.SpawnContext2(ctx, ord)
			xl := xlog.FromContextSafe(ctx2)

			var ffReq FaceReq
			ffReq.Data.IMG = req.Data.IMG
			ffReq.Data.Attribute.Pts = face.Pts
			ff, err := s.EvalFaceFeature(ctx, ffReq)
			if err != nil {
				xl.Errorf("get face feature failed. %v", err)
				return ErrInternal(err.Error())
			}

			var pReq PoliticianReq
			pReq.Data.URI = "data:application/octet-stream;base64," +
				base64.StdEncoding.EncodeToString(ff)
			pReq.Data.Attribute.Pts = face.Pts
			pResp, err := s.EvalPolitician(ctx, pReq)
			if err != nil {
				xl.Errorf("get face match failed. %v", err)
				return ErrInternal(err.Error())
			}

			detail := ConvFaceSearch(ctx, &s.Config.SugConfig, face.Pts, &pResp)
			if detail != nil {
				mux.Lock()
				defer mux.Unlock()
				details = append(details, *detail)
			}

			return nil
		})

	}

	err = waiter.Wait()
	if err != nil {
		return ret, err
	}

	ret = *pimage.MergeDetails(ctx, &s.Config.SugConfig, details)
	return
}

func ConvFaceSearch(ctx context.Context, sugcfg *pimage.SugConfig,
	pts [][2]int, resp *evals.FaceSearchRespV2) *pimage.Detail {

	if len(resp.Result.Confidences) > 0 {
		item := resp.Result.Confidences[0]
		if item.Class == "" {
			return nil
		}

		detail := &pimage.Detail{
			Suggestion: pimage.PASS,
			Label:      item.Class,
			Group:      item.Group,
			Score:      item.Score,
			Detections: []pimage.BoundingBox{
				pimage.BoundingBox{
					Pts:   pts,
					Score: item.Score,
				},
			},
		}

		ok := detail.SetSuggestion(sugcfg)
		if !ok {
			return nil
		}

		return detail
	}

	return nil
}
