package ads

import (
	"context"

	"github.com/go-kit/kit/endpoint"

	"qiniu.com/argus/service/service/biz"
	pimage "qiniu.com/argus/service/service/image"
)

//----------------------------------------------------------------------------//

type SimpleReq struct {
	Data struct {
		// URI string `json:"uri"`
		IMG pimage.Image
	} `json:"data"`
	Params struct {
		Limit int `json:"limit"`
	} `json:"params"`
}

type AdsQrcodeResp struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Result  struct {
		Detections []QrcodeDetection `json:"detections"`
	} `json:"result"`
}

type QrcodeDetection struct {
	Index int      `json:"index"`
	Class string   `json:"class"`
	Pts   [][2]int `json:"pts"`
	Score float32  `json:"score"`
}

type AdsDetectionResp struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Result  struct {
		Detections []AdsDetection `json:"detections"`
	} `json:"result"`
}

type AdsDetection struct {
	Pts   [][2]int `json:"pts"`
	Score float32  `json:"score"`
}

type AdsRecognitionReq struct {
	Data struct {
		IMG       pimage.Image `json:"img"`
		Attribute struct {
			Detections []AdsDetection `json:"detections"`
		} `json:"attribute,omitempty"`
	} `json:"data"`
}

type AdsRecognitionResp struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Result  struct {
		Texts []struct {
			Pts  [][2]int `json:"pts"`
			Text string   `json:"text"`
		} `json:"texts"`
	} `json:"result"`
}

type AdsClassifierReq struct {
	Data struct {
		// Text []string `json:"text"`
		URI string `json:"uri"`
	} `json:"data"`
	Params struct {
		Type []string `json:"type"`
	} `json:"params,omitempty"`
}

type AdsClassifierResp struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Result  struct {
		Ads struct {
			Summary struct {
				Label string  `json:"label"`
				Score float32 `json:"score"`
			} `json:"summary"`
			Confidences []struct {
				Keys  []string `json:"keys"`
				Label string   `json:"label"`
				Score float32  `json:"score"`
			} `json:"confidences"`
		} `json:"ads"`
	} `json:"result"`
}

//----------------------------------------------------------------------------//
type EvalAdsQrcodeService interface {
	EvalAdsQrcode(ctx context.Context, img SimpleReq) (AdsQrcodeResp, error)
}

var _ EvalAdsQrcodeService = EvalAdsQrcodeEndpoints{}

type EvalAdsQrcodeEndpoints struct {
	EvalAdsQrcodeEP endpoint.Endpoint
}

func (ends EvalAdsQrcodeEndpoints) EvalAdsQrcode(
	ctx context.Context, img SimpleReq,
) (AdsQrcodeResp, error) {
	response, err := ends.EvalAdsQrcodeEP(ctx, img)
	if err != nil {
		return AdsQrcodeResp{}, err
	}
	resp := response.(AdsQrcodeResp)
	return resp, nil
}

var EVAL_ADS_QRCODE_CONFIG = biz.EvalModelConfig{
	Image: "reg.qiniu.com/avaprd/ava-eval-barcode_detect:20190115_v1--20190225-v818-private",
	Args: &biz.ModelConfigArgs{
		BatchSize: 1,
	},
	Model: "ava-ads-qrcode/2019011617.tar",
	Type:  biz.EvalRunTypeServing,
}

//----------------------------------------------------------------------------//

type EvalAdsDetectService interface {
	EvalAdsDetect(ctx context.Context, img SimpleReq) (AdsDetectionResp, error)
}

var _ EvalAdsDetectService = EvalAdsDetectEndpoints{}

type EvalAdsDetectEndpoints struct {
	EvalAdsDetectEP endpoint.Endpoint
}

func (ends EvalAdsDetectEndpoints) EvalAdsDetect(
	ctx context.Context, img SimpleReq,
) (AdsDetectionResp, error) {
	response, err := ends.EvalAdsDetectEP(ctx, img)
	if err != nil {
		return AdsDetectionResp{}, err
	}
	resp := response.(AdsDetectionResp)
	return resp, nil
}

var EVAL_ADS_DET_CONFIG = biz.EvalModelConfig{
	Image: "reg.qiniu.com/avaprd/ava-eval-ataraxia-ocr-detection-east-ad:2018121401--20190226-v836-private",
	Model: "ava-ads-detection/201812191508.tar",
	Args: &biz.ModelConfigArgs{
		BatchSize: 1,
	},
	Type: biz.EvalRunTypeServing,
}

//----------------------------------------------------------------------------//
type EvalAdsRecognitionService interface {
	EvalAdsRecognition(ctx context.Context, req AdsRecognitionReq) (AdsRecognitionResp, error)
}

var _ EvalAdsRecognitionService = EvalAdsRecognitionEndpoints{}

type EvalAdsRecognitionEndpoints struct {
	EvalAdsRecognitionEP endpoint.Endpoint
}

func (ends EvalAdsRecognitionEndpoints) EvalAdsRecognition(
	ctx context.Context, req AdsRecognitionReq,
) (AdsRecognitionResp, error) {
	response, err := ends.EvalAdsRecognitionEP(ctx, req)
	if err != nil {
		return AdsRecognitionResp{}, err
	}
	resp := response.(AdsRecognitionResp)
	return resp, nil
}

var EVAL_ADS_RECOGNITION_CONFIG = biz.EvalModelConfig{
	Image: "reg.qiniu.com/avaprd/ava-eval-ataraxia-ocr-recognition-scene:20181221v1.0--20190226-v835-private",
	Model: "ava-ads-recognition/201811261856.tar",
	Args: &biz.ModelConfigArgs{
		BatchSize: 1,
	},
	Type: biz.EvalRunTypeServing,
}

//----------------------------------------------------------------------------//
type EvalAdsClassifierService interface {
	EvalAdsClassifier(ctx context.Context, req AdsClassifierReq) (AdsClassifierResp, error)
}

var _ EvalAdsClassifierService = EvalAdsClassifierEndpoints{}

type EvalAdsClassifierEndpoints struct {
	EvalAdsClassifierEP endpoint.Endpoint
}

func (ends EvalAdsClassifierEndpoints) EvalAdsClassifier(
	ctx context.Context, req AdsClassifierReq,
) (AdsClassifierResp, error) {
	response, err := ends.EvalAdsClassifierEP(ctx, req)
	if err != nil {
		return AdsClassifierResp{}, err
	}
	resp := response.(AdsClassifierResp)
	return resp, nil
}

var EVAL_ADS_CLASSIFIER_CONFIG = biz.EvalModelConfig{
	Image: "reg.qiniu.com/avaprd/ava-eval-ataraxia-ocr-scene-textidentify:20190130v2.0--20190226-v828-private",
	Args: &biz.ModelConfigArgs{
		BatchSize: 1,
	},
	CustomFiles: map[string]string{
		"keyword_file": "ava-ads-classify/201901301850.keyword.csv",
	},
	Type: biz.EvalRunTypeServing,
}
