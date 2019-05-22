package ocrtext

import (
	"context"
	"strings"

	"github.com/go-kit/kit/endpoint"
	xlog "github.com/qiniu/xlog.v1"
	. "qiniu.com/argus/service/service"
	pimage "qiniu.com/argus/service/service/image"
)

const (
	IgnoreImageType string = "others"
	OtherTextScene  string = "other-text"
)

type OcrTextReq struct {
	Data struct {
		IMG pimage.Image
	} `json:"data"`
}

type OcrTextResp struct {
	Code    int           `json:"code"`
	Message string        `json:"message"`
	Result  OcrTextResult `json:"result"`
}

type OcrTextResult struct {
	Type   string      `json:"type"`
	Bboxes [][4][2]int `json:"bboxes"`
	Texts  []string    `json:"texts"`
}

type OcrTextService interface {
	OcrText(ctx context.Context, req OcrTextReq) (ret OcrTextResp, err error)
}

var _ OcrTextService = OcrTextEndpoints{}

type OcrTextEndpoints struct {
	OcrTextEP endpoint.Endpoint
}

func (ends OcrTextEndpoints) OcrText(ctx context.Context, req OcrTextReq) (
	OcrTextResp, error) {
	response, err := ends.OcrTextEP(ctx, req)
	if err != nil {
		return OcrTextResp{}, err
	}
	resp := response.(OcrTextResp)
	return resp, err
}

var _ OcrTextService = ocrTextService{}

type ocrTextService struct {
	eOC  EvalOcrTextClassifyService
	eOD  EvalOcrCtpnService
	eOR  EvalOcrTextRecognizeService
	eOSD EvalOcrSceneDetectService
	eOSR EvalOcrSceneRecognizeService
}

func NewOcrTextService(oc EvalOcrTextClassifyService, od EvalOcrCtpnService, or EvalOcrTextRecognizeService,
	osd EvalOcrSceneDetectService, osr EvalOcrSceneRecognizeService) (OcrTextService, error) {
	return ocrTextService{eOC: oc, eOD: od, eOR: or, eOSD: osd, eOSR: osr}, nil
}

func (s ocrTextService) OcrText(ctx context.Context, args OcrTextReq) (ret OcrTextResp, err error) {
	var xl = xlog.FromContextSafe(ctx)

	ret = OcrTextResp{}
	ret.Result.Bboxes = make([][4][2]int, 0)
	ret.Result.Texts = make([]string, 0)

	if strings.TrimSpace(string(args.Data.IMG.URI)) == "" {
		xl.Error("empty data.uri")
		return OcrTextResp{}, ErrArgs("empty data.uri")
	}

	// ocr text classfy
	var otClassifyReq EvalOcrTextClassifyReq
	otClassifyReq.Data.IMG = args.Data.IMG
	otClassifyResp, err := s.eOC.EvalOcrTextClassify(ctx, otClassifyReq)
	if err != nil {
		xl.Errorf("image ocr text classify failed. %v", err)
		return ret, ErrInternal(err.Error())
	}
	if len(otClassifyResp.Result.Confidences) != 1 {
		xl.Errorf("ocr text classify expect to get one result, but got %d", len(otClassifyResp.Result.Confidences))
		return ret, ErrInternal("wrong classify result")
	}
	ret.Result.Type = otClassifyResp.Result.Confidences[0].Class
	xl.Info("ocr text image_type: ", ret.Result.Type)

	if otClassifyResp.Result.Confidences[0].Class != OtherTextScene && otClassifyResp.Result.Confidences[0].Class != IgnoreImageType {
		// weixin-weibo process
		// ocr text detect
		var (
			otDetectReq  EvalOcrCtpnReq
			otDetectResp EvalOcrCtpnResp
		)
		otDetectReq.Data.IMG = args.Data.IMG
		otDetectResp, err = s.eOD.EvalOcrCtpn(ctx, otDetectReq)
		if err != nil {
			xl.Errorf("image ocr text detect failed. %v", err)
			return ret, ErrInternal(err.Error())
		}
		if len(otDetectResp.Result.Bboxes) == 0 {
			xl.Debugf("ocr text detect got no bbox, ignore this image")
			// return OcrTextResp{}, ErrTextNotFound("found no text")
			return ret, nil
		}
		xl.Infof("ocr text detect got bboxes: %v", otDetectResp.Result.Bboxes)

		//ocr text recognize
		var (
			otRecognizeReq  EvalOcrTextRecognizeReq
			otRecognizeResp EvalOcrTextRecognizeResp
		)
		otRecognizeReq.Data.IMG = args.Data.IMG
		for i := 0; i < len(otDetectResp.Result.Bboxes); i++ {
			otRecognizeReq.Params.Bboxes = append(otRecognizeReq.Params.Bboxes, [4]int{
				otDetectResp.Result.Bboxes[i][0][0],
				otDetectResp.Result.Bboxes[i][0][1],
				otDetectResp.Result.Bboxes[i][2][0],
				otDetectResp.Result.Bboxes[i][2][1],
			})
		}
		otRecognizeReq.Params.ImageType = ret.Result.Type
		otRecognizeResp, err = s.eOR.EvalOcrTextRecognize(ctx, otRecognizeReq)
		if err != nil {
			xl.Errorf("image ocr text recognize failed. %v", err)
			return ret, ErrInternal(err.Error())
		}
		ret.Result.Texts = append(ret.Result.Texts, otRecognizeResp.Result.Texts...)
		ret.Result.Bboxes = append(ret.Result.Bboxes, otRecognizeResp.Result.Bboxes...)
		xl.Infof("ocr text recognize got text: %v", ret.Result.Texts)
	} else {
		// scene process
		// ocr text detect
		var (
			otDetectReq  EvalOcrSceneDetectReq
			otDetectResp EvalOcrSceneDetectResp
		)

		otDetectReq.Data.IMG = args.Data.IMG
		otDetectResp, err = s.eOSD.EvalOcrSceneDetect(ctx, otDetectReq)
		if err != nil {
			xl.Errorf("image ocr text detect failed. %v", err)
			return ret, ErrInternal(err.Error())
		}
		if len(otDetectResp.Result.Bboxes) == 0 {
			xl.Debugf("ocr scene detect got not bbox, ignore this image")
			return ret, nil
		}
		for i := 0; i < len(otDetectResp.Result.Bboxes); i++ {
			ret.Result.Bboxes = append(ret.Result.Bboxes, [4][2]int{
				{otDetectResp.Result.Bboxes[i][0], otDetectResp.Result.Bboxes[i][1]},
				{otDetectResp.Result.Bboxes[i][2], otDetectResp.Result.Bboxes[i][3]},
				{otDetectResp.Result.Bboxes[i][4], otDetectResp.Result.Bboxes[i][5]},
				{otDetectResp.Result.Bboxes[i][6], otDetectResp.Result.Bboxes[i][7]}})
		}

		xl.Infof("ocr scene detect got bboxes: %v", ret.Result.Bboxes)

		//ocr text recognize
		var (
			otRecognizeReq  EvalOcrSceneRecognizeReq
			otRecognizeResp EvalOcrSceneRecognizeResp
		)
		otRecognizeReq.Data.IMG = args.Data.IMG
		otRecognizeReq.Params.Bboxes = otDetectResp.Result.Bboxes
		otRecognizeResp, err = s.eOSR.EvalOcrSceneRecognize(ctx, otRecognizeReq)
		if err != nil {
			xl.Errorf("image ocr text recognize failed. %v", err)
			return ret, ErrInternal(err.Error())
		}
		xl.Infof("ocr text recognize got text: %v", otRecognizeResp.Result.Texts)
		for i := 0; i < len(otRecognizeResp.Result.Texts); i++ {
			ret.Result.Texts = append(ret.Result.Texts, otRecognizeResp.Result.Texts[i].Text)
		}

		xl.Infof("ocr text recognize got text: %v", ret.Result.Texts)
	}

	return
}
