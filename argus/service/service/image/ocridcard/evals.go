package ocridcard

import (
	"context"

	"github.com/go-kit/kit/endpoint"
	pimage "qiniu.com/argus/service/service/image"
)

type EvalOcrSariIdcardDetectReq struct {
	Data struct {
		IMG pimage.Image
	} `json:"data"`
}

type EvalOcrSariIdcardDetectResp struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Result  struct {
		Bboxes [][8]int `json:"bboxes"`
	} `json:"result"`
}

type EvalOcrSariIdcardDetectService interface {
	EvalOcrSariIdcardDetect(ctx context.Context, img EvalOcrSariIdcardDetectReq) (edresp EvalOcrSariIdcardDetectResp, err error)
}

var _ EvalOcrSariIdcardDetectService = EvalOcrSariIdcardDetectEndpoints{}

type EvalOcrSariIdcardDetectEndpoints struct {
	EvalOcrSariIdcardDetectEP endpoint.Endpoint
}

func (ends EvalOcrSariIdcardDetectEndpoints) EvalOcrSariIdcardDetect(ctx context.Context, img EvalOcrSariIdcardDetectReq) (
	EvalOcrSariIdcardDetectResp, error) {
	response, err := ends.EvalOcrSariIdcardDetectEP(ctx, img)
	if err != nil {
		return EvalOcrSariIdcardDetectResp{}, err
	}
	resp := response.(EvalOcrSariIdcardDetectResp)
	return resp, nil
}

//-----------------------------------------------------------------------------------------------------

type EvalOcrSariIdcardRecogReq struct {
	Data struct {
		IMG pimage.Image
	} `json:"data"`
	Params struct {
		Bboxes [][4][2]int `json:"bboxes"`
	} `json:"params"`
}

type EvalOcrSariIdcardRecogResp struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Result  struct {
		Texts []string `json:"text"`
	} `json:"result"`
}

type EvalOcrSariIdcardRecogService interface {
	EvalOcrSariIdcardRecog(ctx context.Context, img EvalOcrSariIdcardRecogReq) (EvalOcrSariIdcardRecogResp, error)
}

var _ EvalOcrSariIdcardRecogService = EvalOcrSariIdcardRecogEndpoints{}

type EvalOcrSariIdcardRecogEndpoints struct {
	EvalOcrSariIdcardRecogEP endpoint.Endpoint
}

func (ends EvalOcrSariIdcardRecogEndpoints) EvalOcrSariIdcardRecog(ctx context.Context, img EvalOcrSariIdcardRecogReq) (
	EvalOcrSariIdcardRecogResp, error) {
	response, err := ends.EvalOcrSariIdcardRecogEP(ctx, img)
	if err != nil {
		return EvalOcrSariIdcardRecogResp{}, err
	}

	resp := response.(EvalOcrSariIdcardRecogResp)
	return resp, err
}

//----------------------------------------------------------------------------------------------------

type EvalOcrSariIdcardPreProcessReq struct {
	Data struct {
		IMG pimage.Image
	} `json:"data"`
	Params struct {
		Type          string      `json:"type"`
		Bboxes        [][4][2]int `json:"bboxes"`
		Class         int         `json:"class"`
		Texts         []string    `json:"texts"`
		Names         []string    `json:"names"`
		Regions       [][4][2]int `json:"regions"`
		DetectedBoxes [][8]int    `json:"detectedBoxes"`
	} `json:"params"`
}

type EvalOcrSariIdcardPreProcessResp struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Result  struct {
		Class         int               `json:"class"`
		AlignedImg    string            `json:"alignedImg"`
		Names         []string          `json:"names"`
		Regions       [][4][2]int       `json:"regions"`
		Bboxes        [][4][2]int       `json:"bboxes"`
		DetectedBoxes [][8]int          `json:"detectedBoxes"`
		Res           map[string]string `json:"res"`
	} `json:"result"`
}

type EvalOcrSariIdcardPreProcessService interface {
	EvalOcrSariIdcardPreProcess(ctx context.Context, img EvalOcrSariIdcardPreProcessReq) (EvalOcrSariIdcardPreProcessResp, error)
}

var _ EvalOcrSariIdcardPreProcessService = EvalOcrSariIdcardPreProcessEndpoints{}

type EvalOcrSariIdcardPreProcessEndpoints struct {
	EvalOcrSariIdcardPreProcessEP endpoint.Endpoint
}

func (end EvalOcrSariIdcardPreProcessEndpoints) EvalOcrSariIdcardPreProcess(ctx context.Context, img EvalOcrSariIdcardPreProcessReq) (
	EvalOcrSariIdcardPreProcessResp, error) {
	response, err := end.EvalOcrSariIdcardPreProcessEP(ctx, img)
	if err != nil {
		return EvalOcrSariIdcardPreProcessResp{}, err
	}

	resp := response.(EvalOcrSariIdcardPreProcessResp)
	return resp, err
}
