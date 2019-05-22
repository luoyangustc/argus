package politician

import (
	"context"
	"encoding/base64"

	"github.com/go-kit/kit/endpoint"
	xlog "github.com/qiniu/xlog.v1"

	. "qiniu.com/argus/service/service"
	pimage "qiniu.com/argus/service/service/image"
	"qiniu.com/argus/utility/evals"
)

type Req struct {
	Data struct {
		IMG pimage.Image
		// URI string `json:"uri"`
	} `json:"data"`
}

// FaceSearchResp ...
type FaceSearchResp struct {
	Code    int              `json:"code"`
	Message string           `json:"message"`
	Result  FaceSearchResult `json:"result"`
}

// FaceSearchResult ...
type FaceSearchResult struct {
	Review     bool               `json:"review"`
	Detections []FaceSearchDetail `json:"detections"`
}

// FaceSearchDetail ...
type FaceSearchDetail struct {
	BoundingBox struct {
		Pts   [][2]int `json:"pts"`
		Score float32  `json:"score"`
	} `json:"boundingBox"`
	Value struct {
		Name   string  `json:"name,omitempty"`
		Group  string  `json:"group,omitempty"`
		Score  float32 `json:"score"`
		Review bool    `json:"review"`
	} `json:"value"`
	Sample *FaceSearchDetailSample `json:"sample,omitempty"`
}

type FaceSearchDetailSample struct {
	URL string   `json:"url"`
	Pts [][2]int `json:"pts"`
}

type FaceSearchService interface {
	FaceSearch(ctx context.Context, img Req) (FaceSearchResp, error)
	PoliticianCensor(ctx context.Context, req pimage.ImageCensorReq) (pimage.SceneResult, error)
}

var _ FaceSearchService = FaceSearchEndpoints{}

type FaceSearchEndpoints struct {
	FaceSearchEP       endpoint.Endpoint
	PoliticianCensorEP endpoint.Endpoint
}

func (ends FaceSearchEndpoints) FaceSearch(ctx context.Context, img Req) (FaceSearchResp, error) {
	response, err := ends.FaceSearchEP(ctx, img)
	if err != nil {
		return FaceSearchResp{}, err
	}
	resp := response.(FaceSearchResp)
	return resp, nil
}

func (ends FaceSearchEndpoints) PoliticianCensor(ctx context.Context, req pimage.ImageCensorReq) (pimage.SceneResult, error) {
	response, err := ends.PoliticianCensorEP(ctx, req)
	if err != nil {
		return pimage.SceneResult{}, err
	}
	resp := response.(pimage.SceneResult)
	return resp, nil
}

var _ FaceSearchService = faceSearchService{}

type Config struct {
	PoliticianThreshold []float32 `json:"politician_threshold"`
	pimage.SugConfig
}

var (
	RuleAlwaysPass = pimage.RuleConfig{
		SureThreshold:    0.42,
		AbandonThreshold: 0.38,
		SureSuggestion:   pimage.PASS,
		UnsureSuggestion: pimage.PASS,
	}
	RuleWithBlock = pimage.RuleConfig{
		SureThreshold:    0.42,
		AbandonThreshold: 0.38,
		SureSuggestion:   pimage.BLOCK,
		UnsureSuggestion: pimage.REVIEW,
	}
	RuleAlwaysReview = pimage.RuleConfig{
		SureThreshold:    0.42,
		AbandonThreshold: 0.38,
		SureSuggestion:   pimage.REVIEW,
		UnsureSuggestion: pimage.REVIEW,
	}
	DEFAULT = Config{
		PoliticianThreshold: []float32{0.38, 0.4, 0.42},
		SugConfig: pimage.SugConfig{
			CensorBy: "group",
			Rules: map[string]pimage.RuleConfig{
				"normal":                 RuleAlwaysPass,
				TAG_AFFAIRS_OFFICIAL_GOV: RuleWithBlock,
				TAG_AFFAIRS_OFFICIAL_ENT: RuleWithBlock,
				TAG_ANTI_CHINA_PEOPLE:    RuleWithBlock,
				TAT_TERRORIST:            RuleWithBlock,
				TAG_AFFAIRS_CELEBRITY:    RuleWithBlock,
				TAG_DOMESTIC_STATESMAN:   RuleAlwaysReview,
				TAG_FOREIGN_STATESMAN:    RuleAlwaysReview,
				TAG_CHINESE_MARTYR:       RuleAlwaysReview,
			},
		},
	}
)

type faceSearchService struct {
	Config
	EvalFaceDetectService
	EvalFaceFeatureService
	EvalPoliticianService
}

func NewFaceSearchService(
	conf Config,
	s1 EvalFaceDetectService,
	s2 EvalFaceFeatureService,
	s3 EvalPoliticianService,
) (FaceSearchService, error) {

	return faceSearchService{
		Config:                 conf,
		EvalFaceDetectService:  s1,
		EvalFaceFeatureService: s2,
		EvalPoliticianService:  s3,
	}, nil
}

//----------------------------------------------------------------------------//
func (s faceSearchService) FaceSearch(
	ctx context.Context, req Req,
) (ret FaceSearchResp, err error) {

	var (
		xl = xlog.FromContextSafe(ctx)
	)

	dResp, err := s.EvalFaceDetect(ctx, req)
	if err != nil {
		xl.Errorf("call facex-detect error: %v", err)
		return ret, ErrInternal(err.Error())
	}
	if dResp.Code != 0 && dResp.Code/100 != 2 {
		xl.Errorf("call facex-detect failed: %d %s", dResp.Code, dResp.Message)
		return ret, ErrInternal(dResp.Message)
	}

	ret.Result.Detections = make([]FaceSearchDetail, 0, len(dResp.Result.Detections))
	for _, face := range dResp.Result.Detections {
		var ff []byte
		var fReq FaceReq
		fReq.Data.IMG = req.Data.IMG
		fReq.Data.Attribute.Pts = face.Pts

		ff, err = s.EvalFaceFeature(ctx, fReq)
		if err != nil {
			xl.Errorf("get face feature failed. %v", err)
			err = ErrInternal(err.Error())
			return
		}

		var mResp evals.FaceSearchRespV2
		var mReq PoliticianReq
		mReq.Data.URI = "data:application/octet-stream;base64," +
			base64.StdEncoding.EncodeToString(ff)
		mReq.Data.Attribute.Pts = face.Pts
		mResp, err = s.EvalPolitician(ctx, mReq)
		if err != nil {
			xl.Errorf("get face match failed. %v", err)
			err = ErrInternal(err.Error())
			return
		}

		if len(mResp.Result.Confidences) == 0 || mResp.Result.Confidences[0].Class == "" {
			continue
		}

		detail := FaceSearchDetail{
			BoundingBox: struct {
				Pts   [][2]int `json:"pts"`
				Score float32  `json:"score"`
			}{
				Pts:   face.Pts,
				Score: face.Score,
			},
		}

		score := mResp.Result.Confidences[0].Score
		if score >= s.Config.PoliticianThreshold[1] {
			detail.Value.Name = mResp.Result.Confidences[0].Class
			detail.Value.Group = mResp.Result.Confidences[0].Group
			detail.Sample = &FaceSearchDetailSample{
				URL: mResp.Result.Confidences[0].Sample.URL,
				Pts: mResp.Result.Confidences[0].Sample.Pts,
			}
		}
		if score > s.Config.PoliticianThreshold[0] &&
			score < s.Config.PoliticianThreshold[2] {
			detail.Value.Review = true
			ret.Result.Review = true
		}
		detail.Value.Score = score
		ret.Result.Detections = append(ret.Result.Detections, detail)
	}

	if len(ret.Result.Detections) == 0 {
		ret.Message = "No valid face info detected"
	}

	return
}
