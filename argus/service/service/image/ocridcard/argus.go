package ocridcard

import (
	"context"
	"strings"

	"github.com/go-kit/kit/endpoint"
	"github.com/qiniu/xlog.v1"
	. "qiniu.com/argus/service/service"
	pimage "qiniu.com/argus/service/service/image"
	simage "qiniu.com/argus/service/service/image"
)

type OcrSariIdcardReq struct {
	Data struct {
		IMG pimage.Image
		// URI string `json:"uri"`
	} `json:"data"`
}

type OcrSariIdcardResp struct {
	Code    int                 `json:"code"`
	Message string              `json:"message"`
	Result  OcrSariIdcardResult `json:"result"`
}

type OcrSariIdcardResult struct {
	URI    string            `json:"uri"`
	Bboxes [][4][2]int       `json:"bboxes"`
	Type   int               `json:"type"`
	Res    map[string]string `json:"res"`
}

type OcrIdcardService interface {
	OcrIdcard(ctx context.Context, req OcrSariIdcardReq) (ret OcrSariIdcardResp, err error)
}

var _ OcrIdcardService = OcrIdcardEndpoints{}

type OcrIdcardEndpoints struct {
	OcrIdcardEP endpoint.Endpoint
}

func (ends OcrIdcardEndpoints) OcrIdcard(ctx context.Context, req OcrSariIdcardReq) (
	OcrSariIdcardResp, error) {
	response, err := ends.OcrIdcardEP(ctx, req)
	if err != nil {
		return OcrSariIdcardResp{}, err
	}

	resp := response.(OcrSariIdcardResp)
	return resp, nil
}

var _ OcrIdcardService = ocrIdcardService{}

type ocrIdcardService struct {
	eOD EvalOcrSariIdcardDetectService
	eOR EvalOcrSariIdcardRecogService
	eOP EvalOcrSariIdcardPreProcessService
}

func NewOcrIdcardService(od EvalOcrSariIdcardDetectService, or EvalOcrSariIdcardRecogService,
	op EvalOcrSariIdcardPreProcessService) (OcrIdcardService, error) {
	return ocrIdcardService{eOD: od, eOP: op, eOR: or}, nil
}

func (s ocrIdcardService) OcrIdcard(ctx context.Context, args OcrSariIdcardReq) (ret OcrSariIdcardResp, err error) {
	var xl = xlog.FromContextSafe(ctx)

	ret = OcrSariIdcardResp{}

	if strings.TrimSpace(string(args.Data.IMG.URI)) == "" {
		xl.Error("empty data.uri")
		return ret, ErrArgs("empty data.uri")
	}

	//pre detect
	var (
		otPreDetectReq  EvalOcrSariIdcardPreProcessReq
		otPreDetectResp EvalOcrSariIdcardPreProcessResp
	)

	otPreDetectReq.Data.IMG = args.Data.IMG
	otPreDetectReq.Params.Type = "predetect"
	otPreDetectResp, err = s.eOP.EvalOcrSariIdcardPreProcess(ctx, otPreDetectReq)

	if err != nil {
		xl.Errorf("id card pre-detect failed. %v", err)
		return ret, ErrInternal(err.Error())
	}
	if len(otPreDetectResp.Result.Bboxes) == 0 {
		xl.Debugf("id card pre-detect got no bbox, ignore this image")
		return ret, nil
	}

	xl.Infof("id card pre-detect got class: %v", otPreDetectResp.Result.Class)
	xl.Infof("id card pre-detect got names: %v", otPreDetectResp.Result.Names)
	xl.Infof("id card pre-detect got bboxes: %v", otPreDetectResp.Result.Bboxes)
	xl.Infof("id card pre-detect got bboxes: %v", otPreDetectResp.Result.Regions)

	//detect
	var (
		otDetectReq  EvalOcrSariIdcardDetectReq
		otDetectResp EvalOcrSariIdcardDetectResp
	)

	otDetectReq.Data.IMG.URI = simage.STRING("data:application/octet-stream;base64," + otPreDetectResp.Result.AlignedImg)
	otDetectResp, err = s.eOD.EvalOcrSariIdcardDetect(ctx, otDetectReq)
	if err != nil {
		xl.Errorf("id card detect failed. %v", err)
		return ret, ErrInternal(err.Error())
	}
	if len(otDetectResp.Result.Bboxes) == 0 {
		xl.Debugf("id card detect got not bbox, ignore this image")
		return ret, nil
	}
	xl.Infof("id card detect got bboxes: %v", otDetectResp.Result.Bboxes)

	//pre recognize
	var (
		otPreRecogReq  EvalOcrSariIdcardPreProcessReq
		otPreRecogResp EvalOcrSariIdcardPreProcessResp
	)
	otPreRecogReq.Data.IMG = otDetectReq.Data.IMG
	otPreRecogReq.Params.Type = "prerecog"
	otPreRecogReq.Params.Class = otPreDetectResp.Result.Class
	otPreRecogReq.Params.DetectedBoxes = otDetectResp.Result.Bboxes
	otPreRecogReq.Params.Names = otPreDetectResp.Result.Names
	otPreRecogReq.Params.Bboxes = otPreDetectResp.Result.Bboxes
	otPreRecogReq.Params.Regions = otPreDetectResp.Result.Regions
	otPreRecogResp, err = s.eOP.EvalOcrSariIdcardPreProcess(ctx, otPreRecogReq)
	if err != nil {
		xl.Errorf("id card pre-recognize failed. %v", err)
		return ret, ErrInternal(err.Error())
	}
	if len(otPreRecogResp.Result.Bboxes) == 0 {
		xl.Debugf("id card pre-recognize got no bbox, ignore this image")
		return ret, nil
	}
	xl.Infof("id card pre-recognize got bboxes: %v", otPreRecogResp.Result.Bboxes)

	//recognize
	var (
		otRecognizeReq  EvalOcrSariIdcardRecogReq
		otRecognizeResp EvalOcrSariIdcardRecogResp
	)
	otRecognizeReq.Data.IMG = otDetectReq.Data.IMG
	otRecognizeReq.Params.Bboxes = otPreRecogResp.Result.Bboxes
	otRecognizeResp, err = s.eOR.EvalOcrSariIdcardRecog(ctx, otRecognizeReq)
	if err != nil {
		xl.Errorf("id card recognize failed. %v", err)
		return ret, ErrInternal(err.Error())
	}
	xl.Infof("id card recognize got texts: %v", otRecognizeResp.Result.Texts)

	//post recognize
	var (
		otPostRecogReq  EvalOcrSariIdcardPreProcessReq
		otPostRecogResp EvalOcrSariIdcardPreProcessResp
	)
	otPostRecogReq.Data.IMG.URI = otDetectReq.Data.IMG.URI
	otPostRecogReq.Params.Type = "postprocess"
	otPostRecogReq.Params.Class = otPreDetectResp.Result.Class
	otPostRecogReq.Params.Bboxes = otPreRecogResp.Result.Bboxes
	otPostRecogReq.Params.Regions = otPreDetectResp.Result.Regions
	otPostRecogReq.Params.Names = otPreDetectResp.Result.Names
	otPostRecogReq.Params.Texts = otRecognizeResp.Result.Texts
	otPostRecogResp, err = s.eOP.EvalOcrSariIdcardPreProcess(ctx, otPostRecogReq)
	if err != nil {
		xl.Errorf("id card post-recognize failed. %v", err)
		return ret, ErrInternal(err.Error())
	}

	xl.Infof("id card post-recognize got result: %v", otPostRecogResp.Result.Res)

	ret.Result.URI = otPreDetectResp.Result.AlignedImg
	ret.Result.Bboxes = otPreRecogResp.Result.Bboxes
	ret.Result.Type = otPreDetectResp.Result.Class
	ret.Result.Res = otPostRecogResp.Result.Res

	// if err == nil && env.UserInfo.Utype != NoChargeUtype {
	// 	setStateHeader(env.W.Header(), "OCR_IDCARD", 1)
	// }
	return

}
