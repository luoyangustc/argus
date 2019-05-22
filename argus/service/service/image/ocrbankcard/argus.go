package ocrbankcard

import (
	"context"
	"strings"

	"github.com/go-kit/kit/endpoint"
	"github.com/qiniu/xlog.v1"
	. "qiniu.com/argus/service/service"
	pimage "qiniu.com/argus/service/service/image"
)

type OcrSariBankcardReq struct {
	Data struct {
		IMG pimage.Image
		// URI string `json:"uri"`
	} `json:"data"`
}

type OcrSariBankcardResp struct {
	Code    int                   `json:"code"`
	Message string                `json:"message"`
	Result  OcrSariBankcardResult `json:"result"`
}

type OcrSariBankcardResult struct {
	Bboxes [][4][2]int       `json:"bboxes"`
	Res    map[string]string `json:"res"`
}

type OcrBankcardService interface {
	OcrBankcard(ctx context.Context, req OcrSariBankcardReq) (ret OcrSariBankcardResp, err error)
}

var _ OcrBankcardService = OcrBankcardEndpoints{}

type OcrBankcardEndpoints struct {
	OcrBankcardEP endpoint.Endpoint
}

func (ends OcrBankcardEndpoints) OcrBankcard(ctx context.Context, req OcrSariBankcardReq) (
	OcrSariBankcardResp, error) {

	response, err := ends.OcrBankcardEP(ctx, req)
	if err != nil {
		return OcrSariBankcardResp{}, err
	}

	resp := response.(OcrSariBankcardResp)
	return resp, nil
}

var _ OcrBankcardService = ocrBankcardService{}

type ocrBankcardService struct {
	eOD EvalOcrSariBankcardDetectService
	eOR EvalOcrSariBankcardRecogService
}

func NewOcrBankcardService(od EvalOcrSariBankcardDetectService, or EvalOcrSariBankcardRecogService) (
	OcrBankcardService, error) {
	return ocrBankcardService{eOD: od, eOR: or}, nil
}

func (s ocrBankcardService) OcrBankcard(ctx context.Context, args OcrSariBankcardReq) (
	ret OcrSariBankcardResp, err error) {
	var xl = xlog.FromContextSafe(ctx)
	ret = OcrSariBankcardResp{}

	if strings.TrimSpace(string(args.Data.IMG.URI)) == "" {
		xl.Error("empty data.uri")
		return ret, ErrArgs("empty data.uri")
	}

	//detect

	var (
		otDetectReq  EvalOcrSariBankcardDetectReq
		otDetectResp EvalOcrSariBankcardDetectResp
	)

	otDetectReq.Data.IMG = args.Data.IMG
	otDetectResp, err = s.eOD.EvalOcrSariBankcardDetect(ctx, otDetectReq)
	if err != nil {
		xl.Errorf("Bank card detect failed. %v", err)
		return ret, ErrInternal(err.Error())
	}
	if len(otDetectResp.Result.Bboxes) == 0 {
		xl.Debugf("Bank card detect got no bbox, ignore this image")
		return ret, nil
	}
	xl.Infof("Bank card detect got bboxes: %v", otDetectResp.Result.Bboxes)

	//recognize
	var (
		otRecognizeReq  EvalOcrSariBankcardRecogReq
		otRecognizeResp EvalOcrSariBankcardRecogResp
	)
	otRecognizeReq.Data.IMG = otDetectReq.Data.IMG
	otRecognizeReq.Params.Bboxes = otDetectResp.Result.Bboxes
	otRecognizeResp, err = s.eOR.EvalOcrSariBankcardRecog(ctx, otRecognizeReq)
	if err != nil {
		xl.Errorf("Bank card recognize failed. %v", err)
		return ret, ErrInternal(err.Error())
	}
	xl.Infof("Bank card recognize got texts: %v", otRecognizeResp.Result.Texts)

	//post process - only derive number and bank name
	num := 0
	ret.Result.Bboxes = make([][4][2]int, 2)
	ret.Result.Res = map[string]string{"开户银行": "", "卡号": ""}
	for index, res := range otRecognizeResp.Result.Texts {
		if strings.Contains(res, "银行") {
			ret.Result.Res["开户银行"] = res
			ret.Result.Bboxes[1] = otDetectResp.Result.Bboxes[index]
			continue
		}

		numres := 0
		for _, i := range res {
			if i >= '0' && i <= '9' {
				numres++
			}
		}

		if numres > num {
			num = numres
			ret.Result.Res["卡号"] = res
			ret.Result.Bboxes[0] = otDetectResp.Result.Bboxes[index]

		}
	}
	xl.Infof("result: %v", ret)

	// if err == nil && env.UserInfo.Utype != NoChargeUtype {
	// 	setStateHeader(env.W.Header(), "OCR_BANKCARD", 1)
	// }
	return

}
