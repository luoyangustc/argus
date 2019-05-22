package ocrvat

import (
	"context"
	"strings"

	"github.com/go-kit/kit/endpoint"
	xlog "github.com/qiniu/xlog.v1"
	. "qiniu.com/argus/service/service"
	pimage "qiniu.com/argus/service/service/image"
)

type OcrSariVatReq struct {
	Data struct {
		IMG pimage.Image
	} `json:"data"`
}

type OcrSariVatResp struct {
	Code    int              `json:"code"`
	Message string           `json:"message"`
	Result  OcrSariVatResult `json:"result"`
}

type OcrSariVatResult struct {
	URI    string                 `json:"uri"`
	Bboxes [][4][2]float32        `json:"bboxes"`
	Res    map[string]interface{} `json:"res"`
}

type OcrSariVatService interface {
	OcrSariVat(ctx context.Context, req OcrSariVatReq) (OcrSariVatResp, error)
}

var _ OcrSariVatService = OcrSariVatEndpoints{}

type OcrSariVatEndpoints struct {
	OcrSariVatEP endpoint.Endpoint
}

func (ends OcrSariVatEndpoints) OcrSariVat(ctx context.Context, req OcrSariVatReq) (OcrSariVatResp, error) {
	response, err := ends.OcrSariVatEP(ctx, req)
	if err != nil {
		return OcrSariVatResp{}, err
	}
	resp := response.(OcrSariVatResp)
	return resp, nil
}

type ocrSariVatService struct {
	EvalOcrSariVatDetectService
	EvalOcrSariVatRecogService
	EvalOcrSariVatPostProcessService
}

func NewOcrSariVatService(vds EvalOcrSariVatDetectService, vrs EvalOcrSariVatRecogService,
	vpps EvalOcrSariVatPostProcessService) (OcrSariVatService, error) {
	return ocrSariVatService{EvalOcrSariVatDetectService: vds, EvalOcrSariVatRecogService: vrs,
		EvalOcrSariVatPostProcessService: vpps}, nil
}

func (s ocrSariVatService) OcrSariVat(ctx context.Context, req OcrSariVatReq) (ret OcrSariVatResp, err error) {

	var (
		xl = xlog.FromContextSafe(ctx)
	)

	if strings.TrimSpace(string(req.Data.IMG.URI)) == "" {
		xl.Error("empty data.uri")
		return ret, ErrArgs("empty data.uri")
	}

	ret = OcrSariVatResp{}

	// detect
	var (
		otDetectReq  EvalOcrSariVatDetectReq
		otDetectResp EvalOcrSariVatDetectResp
	)
	otDetectReq.Data.IMG.URI = req.Data.IMG.URI
	otDetectReq.Params.Type = "detect"
	// xl.Errorf("otDetectReq: %v", otDetectReq)
	resp, err := s.EvalOcrSariVatDetect(ctx, otDetectReq)
	if err != nil {
		xl.Errorf("call EvalOcrSariVatDetect error:%v", err)
		return ret, ErrInternal(err.Error())
	}
	otDetectResp = EvalOcrSariVatDetectResp(resp)
	if len(otDetectResp.Result.Bboxes) == 0 {
		xl.Debugf("vat ticket detect got no bbox, ignore this image")
		return
	}
	xl.Infof("vat ticket detect got bboxes: %v", otDetectResp.Result.Bboxes)

	// recognize
	var (
		otRecognizeReq  EvalOcrSariVatRecogReq
		otRecognizeResp EvalOcrSariVatRecogResp
	)
	otRecognizeReq.Data.IMG.URI = req.Data.IMG.URI
	otRecognizeReq.Params.Bboxes = otDetectResp.Result.Bboxes
	resp2, err := s.EvalOcrSariVatRecog(ctx, otRecognizeReq)
	if err != nil {
		xl.Errorf("vat ticket recognize failed. %v", err)
		return
	}
	otRecognizeResp = EvalOcrSariVatRecogResp(resp2)
	xl.Infof("vat ticket recognize got texts: %v", otRecognizeResp.Result.Texts)

	// post recognize
	var (
		otPostRecogReq  EvalOcrSariVatPostProcessReq
		otPostRecogResp EvalOcrSariVatPostProcessResp
	)
	otPostRecogReq.Data.IMG.URI = req.Data.IMG.URI
	otPostRecogReq.Params.Type = "postrecog"
	otPostRecogReq.Params.Texts = otRecognizeResp.Result.Texts
	resp3, err := s.EvalOcrSariVatPostProcess(ctx, otPostRecogReq)
	if err != nil {
		xl.Errorf("vat ticket post-recognize failed. %v", err)
		return
	}
	otPostRecogResp = EvalOcrSariVatPostProcessResp(resp3)
	xl.Infof("vat ticket post-recognize got result: %v", otPostRecogResp.Result)

	ret.Result.Bboxes = otDetectResp.Result.Bboxes
	ret.Result.Res = otPostRecogResp.Result

	return
}
