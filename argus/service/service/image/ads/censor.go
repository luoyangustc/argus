package ads

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"github.com/go-kit/kit/endpoint"
	xlog "github.com/qiniu/xlog.v1"

	. "qiniu.com/argus/service/service"
	pimage "qiniu.com/argus/service/service/image"
)

const ADS = "ads"

type AdsService interface {
	AdsCensor(ctx context.Context, args pimage.ImageCensorReq) (resp pimage.SceneResult, err error)
}

var _ AdsService = AdsEndpoints{}

type AdsEndpoints struct {
	AdsCensorEP endpoint.Endpoint
}

func (ends AdsEndpoints) AdsCensor(ctx context.Context, args pimage.ImageCensorReq) (resp pimage.SceneResult, err error) {
	response, err := ends.AdsCensorEP(ctx, args)
	if err != nil {
		return pimage.SceneResult{}, err
	}
	resp = response.(pimage.SceneResult)
	return resp, nil
}

var _ AdsService = adsService{}

type Config struct {
	pimage.SugConfig
}

var (
	RuleAbandon = pimage.RuleConfig{
		SureThreshold:    1.1,
		AbandonThreshold: 1.1,
		SureSuggestion:   pimage.PASS,
		UnsureSuggestion: pimage.PASS,
	}
	RuleWithBlock = pimage.RuleConfig{
		SureThreshold:    0.85,
		AbandonThreshold: 0.3,
		SureSuggestion:   pimage.BLOCK,
		UnsureSuggestion: pimage.REVIEW,
	}

	DEFAULT = Config{
		SugConfig: pimage.SugConfig{
			CensorBy: "label",
			Rules: map[string]pimage.RuleConfig{
				"normal":      RuleAbandon,
				"ads":         RuleWithBlock,
				"summary_ads": RuleWithBlock,
				"qr_code":     RuleWithBlock,
				"bar_code":    RuleWithBlock,
			},
		},
	}
)

type adsService struct {
	Config
	EvalAdsQrcodeService
	EvalAdsDetectService
	EvalAdsRecognitionService
	EvalAdsClassifierService
}

func NewAdsService(
	conf Config,
	s1 EvalAdsQrcodeService,
	s2 EvalAdsDetectService,
	s3 EvalAdsRecognitionService,
	s4 EvalAdsClassifierService,
) (AdsService, error) {
	return adsService{
		Config:                    conf,
		EvalAdsQrcodeService:      s1,
		EvalAdsDetectService:      s2,
		EvalAdsRecognitionService: s3,
		EvalAdsClassifierService:  s4,
	}, nil
}

func (s adsService) AdsCensor(ctx context.Context, args pimage.ImageCensorReq) (pimage.SceneResult, error) {
	req := SimpleReq{
		Data: struct {
			// URI string `json:"uri"`
			IMG pimage.Image
		}{
			IMG: args.Data.IMG,
		},
	}

	var (
		sugcfg = &s.Config.SugConfig
		xl     = xlog.FromContextSafe(ctx)
	)

	qrResp, qerr := s.EvalAdsQrcode(ctx, req)
	qrSr := ConvAdsQrcode(ctx, sugcfg, &qrResp)
	if qerr == nil &&
		len(qrResp.Result.Detections) != 0 &&
		qrSr.Suggestion == pimage.BLOCK {
		// 模型有明确结果，二维码/条形码达到block则直接返回
		return *qrSr, nil
	}

	resp, derr := s.EvalAdsDetect(ctx, req)

	if derr != nil {
		derr = ErrInternal(derr.Error())
		xl.Errorf("call ads_detect failed. %v", derr)
		return pimage.SceneResult{}, derr
	}

	if len(resp.Result.Detections) == 0 {

		return pimage.SceneResult{Suggestion: pimage.PASS}, nil
	}

	req2 := AdsRecognitionReq{
		Data: struct {
			IMG       pimage.Image `json:"img"`
			Attribute struct {
				Detections []AdsDetection `json:"detections"`
			} `json:"attribute,omitempty"`
		}{
			IMG: req.Data.IMG,
			Attribute: struct {
				Detections []AdsDetection `json:"detections"`
			}{
				Detections: resp.Result.Detections,
			},
		},
	}

	resp2, rerr := s.EvalAdsRecognition(ctx, req2)
	if rerr != nil {
		rerr = ErrInternal(rerr.Error())
		return pimage.SceneResult{}, rerr
	}

	if len(resp2.Result.Texts) == 0 {
		return pimage.SceneResult{Suggestion: pimage.PASS}, nil
	}

	req3 := AdsClassifierReq{
		Params: struct {
			Type []string `json:"type"`
		}{
			Type: []string{ADS},
		},
	}

	var Texts []string
	for _, txt := range resp2.Result.Texts {
		Texts = append(Texts, txt.Text) //兼容serving——eval 接口s
	}
	bytes, _ := json.Marshal(Texts)
	base64uri := base64.StdEncoding.EncodeToString(bytes)

	uri := fmt.Sprintf("%s,%s", "data:application/octet-stream;base64", string(base64uri))
	req3.Data.URI = uri
	resp3, cerr := s.EvalAdsClassifier(ctx, req3)
	if cerr != nil {
		cerr = ErrInternal(cerr.Error())
		return pimage.SceneResult{}, cerr
	}
	if len(resp3.Result.Ads.Confidences) == 0 {
		return pimage.SceneResult{Suggestion: pimage.PASS}, nil
	}
	sr := ConvAdsClassifier(ctx, sugcfg, &resp2, &resp3)
	sr2 := pimage.MergeDetails(ctx, sugcfg, sr.Details)
	return *sr2, nil
}

func ConvAdsQrcode(ctx context.Context, sugcfg *pimage.SugConfig,
	resp *AdsQrcodeResp) *pimage.SceneResult {

	sceneResult := pimage.SceneResult{
		Suggestion: pimage.PASS,
	}

	if resp == nil {
		return &sceneResult
	}

	for _, item := range resp.Result.Detections {
		if item.Class == "" {
			continue
		}

		detail := pimage.Detail{
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

		sceneResult.Details = append(sceneResult.Details, detail)
		sceneResult.Suggestion = sceneResult.Suggestion.Update(detail.Suggestion)
	}

	return &sceneResult
}

func ConvAdsClassifier(ctx context.Context, sugcfg *pimage.SugConfig,
	resp *AdsRecognitionResp, resp2 *AdsClassifierResp) *pimage.SceneResult {

	sceneResult := pimage.SceneResult{
		Suggestion: pimage.PASS,
	}

	if resp == nil || resp2 == nil ||
		len(resp.Result.Texts) != len(resp2.Result.Ads.Confidences) {
		return &sceneResult
	}

	for i, item := range resp2.Result.Ads.Confidences {
		if item.Label == "" {
			continue
		}

		// 关键词扔进comments
		sort.Sort(Keywords(item.Keys))
		comments := strings.Join(item.Keys, ";")

		detail := pimage.Detail{
			Suggestion: pimage.PASS,
			Label:      item.Label,
			Score:      item.Score,
			Comments:   comments,
			Detections: []pimage.BoundingBox{
				pimage.BoundingBox{
					Pts:      resp.Result.Texts[i].Pts,
					Score:    item.Score,
					Comments: comments,
				},
			},
		}

		ok := detail.SetSuggestion(sugcfg)
		if !ok {
			continue
		}
		sceneResult.Details = append(sceneResult.Details, detail)
		sceneResult.Suggestion = sceneResult.Suggestion.Update(detail.Suggestion)

	}

	// summary 当作标签处理
	summaryDetail := pimage.Detail{
		Suggestion: pimage.PASS,
		Label:      fmt.Sprintf("summary_%s", resp2.Result.Ads.Summary.Label),
		Score:      resp2.Result.Ads.Summary.Score,
	}
	ok := summaryDetail.SetSuggestion(sugcfg)
	if ok {
		sceneResult.Details = append(sceneResult.Details, summaryDetail)
		sceneResult.Suggestion = sceneResult.Suggestion.Update(summaryDetail.Suggestion)
	}

	return &sceneResult
}

type Keywords []string

func (kws Keywords) Len() int {
	return len(kws)
}

func (kws Keywords) Less(i, j int) bool {
	return kws[i] < kws[j]
}

func (kws Keywords) Swap(i, j int) {
	kws[i], kws[j] = kws[j], kws[i]
}
