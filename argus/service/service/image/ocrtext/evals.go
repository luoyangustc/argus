package ocrtext

import (
	"context"

	"github.com/go-kit/kit/endpoint"
	pimage "qiniu.com/argus/service/service/image"
)

//------------------------------------------------------------------------------
//文本分类
type EvalOcrTextClassifyReq struct {
	Data struct {
		IMG pimage.Image
	} `json:"data"`
}

type EvalOcrTextClassifyResp struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Result  struct {
		Confidences []struct {
			Index int     `json:"index"`
			Class string  `json:"class"`
			Score float64 `json:"score"`
		} `json:"confidences"`
	} `json:"result"`
}

type EvalOcrTextClassifyService interface {
	EvalOcrTextClassify(ctx context.Context, img EvalOcrTextClassifyReq) (etcresp EvalOcrTextClassifyResp, err error)
}

var _ EvalOcrTextClassifyService = EvalOcrTextClassifyEndpoints{}

type EvalOcrTextClassifyEndpoints struct {
	EvalOcrTextClassifyEP endpoint.Endpoint
}

func (ends EvalOcrTextClassifyEndpoints) EvalOcrTextClassify(
	ctx context.Context, img EvalOcrTextClassifyReq) (EvalOcrTextClassifyResp, error) {
	response, err := ends.EvalOcrTextClassifyEP(ctx, img)
	if err != nil {
		return EvalOcrTextClassifyResp{}, err
	}
	resp := response.(EvalOcrTextClassifyResp)
	return resp, nil
}

//-------------------------------------------------------------------------------
//长文本检测

type EvalOcrCtpnReq struct {
	Data struct {
		IMG pimage.Image
	} `json:"data"`
}

type EvalOcrCtpnResp struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Result  struct {
		Bboxes [][4][2]int `json:"bboxes"`
	} `json:"result"`
}

type EvalOcrCtpnService interface {
	EvalOcrCtpn(ctx context.Context, img EvalOcrCtpnReq) (EvalOcrCtpnResp, error)
}

var _ EvalOcrCtpnService = EvalOcrCtpnEndpoints{}

type EvalOcrCtpnEndpoints struct {
	EvalOcrCtpnEP endpoint.Endpoint
}

func (ends EvalOcrCtpnEndpoints) EvalOcrCtpn(ctx context.Context, img EvalOcrCtpnReq) (
	EvalOcrCtpnResp, error) {
	response, err := ends.EvalOcrCtpnEP(ctx, img)
	if err != nil {
		return EvalOcrCtpnResp{}, err
	}
	resp := response.(EvalOcrCtpnResp)
	return resp, err
}

//-----------------------------------------------------------------------------
//长文本识别
type EvalOcrTextRecognizeReq struct {
	Data struct {
		IMG pimage.Image
	} `json:"data"`
	Params struct {
		Bboxes    [][4]int `json:"bboxes"`
		ImageType string   `json:"image_type"`
	} `json:"params"`
}

type EvalOcrTextRecognizeResp struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Result  struct {
		Bboxes [][4][2]int `json:"bboxes"`
		Texts  []string    `json:"texts"`
	} `json:"result"`
}

type EvalOcrTextRecognizeService interface {
	EvalOcrTextRecognize(ctx context.Context, img EvalOcrTextRecognizeReq) (EvalOcrTextRecognizeResp, error)
}

var _ EvalOcrTextRecognizeService = EvalOcrTextRecognizeEndpoints{}

type EvalOcrTextRecognizeEndpoints struct {
	EvalOcrTextRecognizeEP endpoint.Endpoint
}

func (ends EvalOcrTextRecognizeEndpoints) EvalOcrTextRecognize(ctx context.Context, img EvalOcrTextRecognizeReq) (
	EvalOcrTextRecognizeResp, error) {
	response, err := ends.EvalOcrTextRecognizeEP(ctx, img)
	if err != nil {
		return EvalOcrTextRecognizeResp{}, err
	}
	resp := response.(EvalOcrTextRecognizeResp)
	return resp, err
}

//---------------------------------------------------------------------------
//短文本检测

type EvalOcrSceneDetectReq struct {
	Data struct {
		IMG pimage.Image
	} `json:"data"`
}

type EvalOcrSceneDetectResp struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Result  struct {
		Bboxes [][8]int `json:"bboxes"`
	} `json:"result"`
}

type EvalOcrSceneDetectService interface {
	EvalOcrSceneDetect(ctx context.Context, img EvalOcrSceneDetectReq) (EvalOcrSceneDetectResp, error)
}

var _ EvalOcrSceneDetectService = EvalOcrSceneDetectEndpoints{}

type EvalOcrSceneDetectEndpoints struct {
	EvalOcrSceneDetectEP endpoint.Endpoint
}

func (ends EvalOcrSceneDetectEndpoints) EvalOcrSceneDetect(ctx context.Context, img EvalOcrSceneDetectReq) (
	EvalOcrSceneDetectResp, error) {
	response, err := ends.EvalOcrSceneDetectEP(ctx, img)
	if err != nil {
		return EvalOcrSceneDetectResp{}, err
	}
	resp := response.(EvalOcrSceneDetectResp)
	return resp, err
}

//--------------------------------------------------------------------------------------
//短文本识别

type EvalOcrSceneRecognizeReq struct {
	Data struct {
		IMG pimage.Image
	} `json:"data"`
	Params struct {
		Bboxes [][8]int `json:"bboxes"`
	} `json:"params"`
}

type EvalOcrSceneRecognizeResp struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Result  struct {
		Texts []OcrSceneRespResult `json:"texts"`
	} `json:"result"`
}

type OcrSceneRespResult struct {
	Bboxes [8]int `json:"bboxes"`
	Text   string `json:"text"`
}

type EvalOcrSceneRecognizeService interface {
	EvalOcrSceneRecognize(ctx context.Context, img EvalOcrSceneRecognizeReq) (EvalOcrSceneRecognizeResp, error)
}

var _ EvalOcrSceneRecognizeService = EvalOcrSceneRecognizeEndpoints{}

type EvalOcrSceneRecognizeEndpoints struct {
	EvalOcrSceneRecognizeEP endpoint.Endpoint
}

func (ends EvalOcrSceneRecognizeEndpoints) EvalOcrSceneRecognize(ctx context.Context, img EvalOcrSceneRecognizeReq) (
	EvalOcrSceneRecognizeResp, error) {
	response, err := ends.EvalOcrSceneRecognizeEP(ctx, img)
	if err != nil {
		return EvalOcrSceneRecognizeResp{}, err
	}
	resp := response.(EvalOcrSceneRecognizeResp)
	return resp, err
}
