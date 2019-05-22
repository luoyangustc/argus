package politician

import (
	"context"

	"github.com/go-kit/kit/endpoint"

	"qiniu.com/argus/service/service/biz"
	pimage "qiniu.com/argus/service/service/image"
	"qiniu.com/argus/utility/evals"
)

//----------------------------------------------------------------------------//
type EvalFaceDetectService interface {
	EvalFaceDetect(ctx context.Context, img Req) (evals.FaceDetectResp, error)
}

var _ EvalFaceDetectService = EvalFaceDetectEndpoints{}

type EvalFaceDetectEndpoints struct {
	EvalFaceDetectEP endpoint.Endpoint
}

func (ends EvalFaceDetectEndpoints) EvalFaceDetect(ctx context.Context, img Req) (evals.FaceDetectResp, error) {
	response, err := ends.EvalFaceDetectEP(ctx, img)
	if err != nil {
		return evals.FaceDetectResp{}, err
	}
	resp := response.(evals.FaceDetectResp)
	return resp, nil
}

var EVAL_FACE_DET_CONFIG = biz.EvalModelConfig{
	Image: "reg.qiniu.com/avaprd/aisdk-face_detect:20190108-v105-dev",
	Model: "ava-facex-detect/tron-refinenet-mtcnn/20180824-private.tar",
	Args: &biz.ModelConfigArgs{
		BatchSize: 4,
		CustomValues: map[string]interface{}{
			"gpu_id":               0,
			"const_use_quality":    0,
			"output_quality_score": 1,
			"min_face":             50,
			"frontend_fd":          "ipc://frontend_fd.ipc",
			"backend_fd":           "ipc://backend_fd.ipc",
			"frontend_qa":          "ipc://frontend_qa.ipc",
			"backend_qa":           "ipc://backend_qa.ipc",
		},
	},
	Type: biz.EvalRunTypeServing,
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

func (ends EvalFaceFeatureEndpoints) EvalFaceFeature(ctx context.Context, img FaceReq) ([]byte, error) {
	response, err := ends.EvalFaceFeatureEP(ctx, img)
	if err != nil {
		return []byte{}, err
	}
	resp := response.([]byte)
	return resp, nil
}

var EVAL_FACE_FEATURE_V4_CONFIG = biz.EvalModelConfig{
	Image: "reg.qiniu.com/avaprd/aisdk-face_feature:20190215-v257-dev",
	Model: "ava-facex-feature-v4/caffe-mxnet/201811051030.tar",
	Args: &biz.ModelConfigArgs{
		BatchSize: 4,
		CustomValues: map[string]interface{}{
			"gpu_id":               0,
			"mirror_trick":         0,
			"feature_output_layer": "fc1",
			"min_face_size":        50,
		},
	},
	Type: biz.EvalRunTypeServing,
}

//----------------------------------------------------------------------------//
type PoliticianReq struct {
	Data struct {
		URI       string `json:"uri"`
		Attribute struct {
			Pts [][2]int `json:"pts"`
		} `json:"attribute,omitempty"`
	} `json:"data"`
	Params struct {
		Limit int `json:"limit,omitempty"`
	} `json:"params,omitempty"`
}

type EvalPoliticianService interface {
	EvalPolitician(ctx context.Context, img PoliticianReq) (evals.FaceSearchRespV2, error)
}

var _ EvalPoliticianService = EvalPoliticianEndpoints{}

type EvalPoliticianEndpoints struct {
	EvalPoliticianEP endpoint.Endpoint
}

func (ends EvalPoliticianEndpoints) EvalPolitician(ctx context.Context, img PoliticianReq) (evals.FaceSearchRespV2, error) {
	response, err := ends.EvalPoliticianEP(ctx, img)
	if err != nil {
		return evals.FaceSearchRespV2{}, err
	}
	resp := response.(evals.FaceSearchRespV2)
	return resp, nil
}

var EVAL_POLITICIAN_CONFIG = biz.EvalModelConfig{
	Image: "reg.qiniu.com/avaprd/aisdk-face_search:20190108-v105-dev",
	CustomFiles: map[string]string{
		"features.line": "ava-politician/other-face-search/20190307.features.line",
		"labels.line":   "ava-politician/other-face-search/20190307.labels.line",
	},
	Type: biz.EvalRunTypeServing,
}
