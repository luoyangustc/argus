package face

import (
	"context"

	"github.com/go-kit/kit/endpoint"
	pimage "qiniu.com/argus/service/service/image"
	"qiniu.com/argus/utility/evals"
)

//----------------------------------------------------------------------------//
type FaceDetecReq struct {
	Data struct {
		IMG pimage.Image
		// URI string `json:"uri"`
	} `json:"data"`
	Params struct {
		UseQuality int `json:"use_quality"`
	} `json:"params"`
}

type EvalFaceDetectService interface {
	EvalFaceDetect(ctx context.Context, img FaceDetecReq) (evals.FaceDetectResp, error)
}

var _ EvalFaceDetectService = EvalFaceDetectEndpoints{}

type EvalFaceDetectEndpoints struct {
	EvalFaceDetectEP endpoint.Endpoint
}

func (ends EvalFaceDetectEndpoints,
) EvalFaceDetect(ctx context.Context, img FaceDetecReq) (evals.FaceDetectResp, error) {
	response, err := ends.EvalFaceDetectEP(ctx, img)
	if err != nil {
		return evals.FaceDetectResp{}, err
	}
	resp := response.(evals.FaceDetectResp)
	return resp, nil
}

//----------------------------------------------------------------------------//

type FaceReq struct {
	Data struct {
		IMG pimage.Image
		// URI       string `json:"uri"`
		Attribute struct {
			Pts [][2]int `json:"pts"`
		} `json:"attribute,omitempty"`
	} `json:"data"`
}

type EvalFaceFeatureService interface {
	EvalFaceFeature(ctx context.Context, img FaceReq) ([]byte, error)
}

var _ EvalFaceFeatureService = EvalFaceFeatureEndpoints{}

type EvalFaceFeatureEndpoints struct {
	EvalFaceFeatureEP endpoint.Endpoint
}

func (ends EvalFaceFeatureEndpoints,
) EvalFaceFeature(ctx context.Context, img FaceReq) ([]byte, error) {
	response, err := ends.EvalFaceFeatureEP(ctx, img)
	if err != nil {
		return []byte{}, err
	}
	resp := response.([]byte)
	return resp, nil
}

//----------------------------------------------------------------------------//
type EvalFaceAttributeService interface {
	EvalFaceAttribute(context.Context, FaceAttributeReq) (FaceAttributeResp, error)
}

var _ EvalFaceAttributeService = EvalFaceAttributeEndpoints{}

type EvalFaceAttributeEndpoints struct {
	EvalFaceAttributeEP endpoint.Endpoint
}

func (ends EvalFaceAttributeEndpoints) EvalFaceAttribute(ctx context.Context, req FaceAttributeReq) (FaceAttributeResp, error) {
	resp, err := ends.EvalFaceAttributeEP(ctx, req)
	if err != nil {
		return FaceAttributeResp{}, err
	}
	return resp.(FaceAttributeResp), nil
}
