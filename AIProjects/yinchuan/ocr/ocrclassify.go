package ocr

import (
	"context"
	"strings"

	"github.com/go-kit/kit/endpoint"
	xlog "github.com/qiniu/xlog.v1"
	. "qiniu.com/argus/service/service"
	pimage "qiniu.com/argus/service/service/image"
)

type OcrClasssifyReq struct {
	Data struct {
		IMG pimage.Image
	} `json:"data"`
}

type OcrClassifyResp struct {
	Code    int                 `json:"code"`
	Message string              `json:"message"`
	Result  []OcrClassifyResult `json:"result"`
}

type OcrClassifyResult struct {
	Bboxes [4][2]int `json:"bboxes"`
	Class  string    `json:"class"`
	Index  int       `json:"index"`
	Score  float32   `json:"score"`
}

type OcrClassifyService interface {
	OcrClassify(ctx context.Context, req OcrClasssifyReq) (ret OcrClassifyResp, err error)
}

var _ OcrClassifyService = OcrClassifyEndpoints{}

type OcrClassifyEndpoints struct {
	OcrClassifyEP endpoint.Endpoint
}

func (ends OcrClassifyEndpoints) OcrClassify(ctx context.Context, req OcrClasssifyReq) (
	OcrClassifyResp, error) {
	response, err := ends.OcrClassifyEP(ctx, req)
	if err != nil {
		return OcrClassifyResp{}, err
	}
	resp := response.(OcrClassifyResp)
	return resp, err
}

var _ OcrClassifyService = ocrClassifyService{}

type ocrClassifyService struct {
	eOC  EvalOcrClassifyService
	eOTC EvalOcrTerrorService
}

func NewOcrClassifyService(oc EvalOcrClassifyService, otc EvalOcrTerrorService) (OcrClassifyService, error) {
	return ocrClassifyService{eOC: oc, eOTC: otc}, nil
}

func (s ocrClassifyService) OcrClassify(ctx context.Context, args OcrClasssifyReq) (ret OcrClassifyResp, err error) {
	var xl = xlog.FromContextSafe(ctx)

	ret = OcrClassifyResp{}
	if strings.TrimSpace(string(args.Data.IMG.URI)) == "" {
		xl.Error("empty data.uri")
		return ret, ErrArgs("empty data.uri")
	}

	//detect
	var (
		otRefinedetReq  EvalOcrRefinedetReq
		otRefinedetResp EvalOcrRefinedetResp

		otTerrorReq  EvalOcrTerrorReq
		otTerrorResp EvalOcrTerrorResp
	)
	otRefinedetReq.Data.IMG = args.Data.IMG
	otTerrorReq.Data.IMG = args.Data.IMG

	otRefinedetResp, err = s.eOC.EvalOcrClassify(ctx, otRefinedetReq)

	if err != nil {
		xl.Errorf("image ocr classify detect failed. %v", err)
		return ret, ErrInternal(err.Error())
	}

	xl.Infof("Image detect got items: %v", otRefinedetResp.Result)

	if 0 < len(otRefinedetResp.Result.Items) && "others" != otRefinedetResp.Result.Items[0].Class {
		ret.Result = otRefinedetResp.Result.Items
		return
	} else {
		otTerrorResp, err = s.eOTC.EvalOcrTerror(ctx, otTerrorReq)
		if err != nil {
			xl.Infof("image ocr further classify detect failed. %v", err)
			return
		}

		ret.Result = append(ret.Result, OcrClassifyResult{
			Class: otTerrorResp.Result.Confidences[0].Class,
			Index: otTerrorResp.Result.Confidences[0].Index,
			Score: otTerrorResp.Result.Confidences[0].Score,
			// Bboxes: [4][2]int{},
		})
		return
	}
}
