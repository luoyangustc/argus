package image_sync

import (
	"context"
	"encoding/json"

	"github.com/go-kit/kit/endpoint"
	"github.com/imdario/mergo"

	"qiniu.com/argus/service/middleware"
	sbiz "qiniu.com/argus/service/scenario/biz"
	scenario "qiniu.com/argus/service/scenario/image_sync"
	"qiniu.com/argus/service/service/image/ads"
)

const (
	VERSION                   = "1.0.0"
	EVAL_ADS_QRCODE_NAME      = "evalAdsQrcode"
	EVAL_ADS_DETECT_NAME      = "evalAdsDetect"
	EVAL_ADS_RECOGNITION_NAME = "evalAdsRecognition"
	EVAL_ADS_CLASSIFIER_NAME  = "evalAdsClassifier"
)

var ON bool = false

func Import(serviceID string) func(interface{}) {
	ON = true
	return func(is interface{}) {
		Init(is.(scenario.ImageServer), serviceID)
	}
}

type Config ads.Config

func (c *Config) UnmarshalJSON(b []byte) error {
	type C Config
	c2 := C{}
	if err := json.Unmarshal(b, &c2); err != nil {
		return err
	}
	c1 := C(ads.DEFAULT)
	_ = mergo.Merge(&c2, &c1)
	*c = Config(c2)
	return nil
}

var (
	Set   scenario.ServiceSetter
	ESet1 sbiz.ServiceEvalSetter
	ESet2 sbiz.ServiceEvalSetter
	ESet3 sbiz.ServiceEvalSetter
	ESet4 sbiz.ServiceEvalSetter

	// 两卡下ads的一种部署方式
	DeployMode2GPUCard = [][]sbiz.ServiceEvalDeployProcess{
		{
			// 其他服务 GPU0
		},
		{
			{Name: EVAL_ADS_QRCODE_NAME, Num: 1},      // ads-qrcode GPU1
			{Name: EVAL_ADS_DETECT_NAME, Num: 1},      // ads-detect GPU1
			{Name: EVAL_ADS_RECOGNITION_NAME, Num: 1}, // ads-recognition GPU1
		},
	}

	// 四卡下ads的一种部署方式
	DeployMode4GPUCard = [][]sbiz.ServiceEvalDeployProcess{
		{
			// 其他服务 GPU0
		},
		{
			{Name: EVAL_ADS_QRCODE_NAME, Num: 1},      // ads-qrcode GPU1
			{Name: EVAL_ADS_DETECT_NAME, Num: 1},      // ads-detect GPU1
			{Name: EVAL_ADS_RECOGNITION_NAME, Num: 1}, // ads-recognition GPU1
		},
		{
			// 其他服务 GPU2
		},
		{
			{Name: EVAL_ADS_QRCODE_NAME, Num: 1},      // ads-qrcode GPU3
			{Name: EVAL_ADS_DETECT_NAME, Num: 1},      // ads-detect GPU3
			{Name: EVAL_ADS_RECOGNITION_NAME, Num: 1}, // ads-recognition GPU3
		},
	}
)

func Init(is scenario.ImageServer, serviceID string) {
	var config = Config(ads.DEFAULT)

	var ts1 ads.EvalAdsQrcodeService
	var ts2 ads.EvalAdsDetectService
	var ts3 ads.EvalAdsRecognitionService
	var ts4 ads.EvalAdsClassifierService

	Set = is.Register(
		scenario.ServiceInfo{ID: serviceID, Name: "ads", Version: VERSION},
		&middleware.ServiceFactory{
			New: func() middleware.Service {
				{
					ss1, _ := ESet1.Gen()
					ts1 = ss1.(ads.EvalAdsQrcodeService)

					ss2, _ := ESet2.Gen()
					ts2 = ss2.(ads.EvalAdsDetectService)

					ss3, _ := ESet3.Gen()
					ts3 = ss3.(ads.EvalAdsRecognitionService)

					ss4, _ := ESet4.Gen()
					ts4 = ss4.(ads.EvalAdsClassifierService)
				}
				s, _ := ads.NewAdsService(ads.Config(config), ts1, ts2, ts3, ts4)
				return s
			},
			NewShell: func() middleware.ServiceEndpoints { return ads.AdsEndpoints{} },
		})

	_ = Set.GetConfig(context.Background(), &config)

	ESet1 = Set.NewEval(
		sbiz.ServiceEvalInfo{Name: EVAL_ADS_QRCODE_NAME, Version: "1.0.0"},
		AdsQrcodeEvalClient,
		func() middleware.ServiceEndpoints { return ads.EvalAdsQrcodeEndpoints{} },
	).SetModel(ads.EVAL_ADS_QRCODE_CONFIG).GenId()

	ESet2 = Set.NewEval(
		sbiz.ServiceEvalInfo{Name: EVAL_ADS_DETECT_NAME, Version: "1.0.0"},
		AdsDetectEvalClient,
		func() middleware.ServiceEndpoints { return ads.EvalAdsDetectEndpoints{} },
	).SetModel(ads.EVAL_ADS_DET_CONFIG).GenId()

	ESet3 = Set.NewEval(
		sbiz.ServiceEvalInfo{Name: EVAL_ADS_RECOGNITION_NAME, Version: "1.0.0"},
		AdsRecognitionEvalClient,
		func() middleware.ServiceEndpoints { return ads.EvalAdsRecognitionEndpoints{} },
	).SetModel(ads.EVAL_ADS_RECOGNITION_CONFIG).GenId()

	ESet4 = Set.NewEval(
		sbiz.ServiceEvalInfo{Name: EVAL_ADS_CLASSIFIER_NAME, Version: "1.0.0"},
		AdsClassifierEvalClient,
		func() middleware.ServiceEndpoints { return ads.EvalAdsClassifierEndpoints{} },
	).SetModel(ads.EVAL_ADS_CLASSIFIER_CONFIG).GenId()

}

type Req struct {
	Data struct {
		URI string `json:"uri"`
	} `json:"data"`
	Params struct {
		Limit int `json:"limit"`
	} `json:"params,omitempty"`
}

func AdsQrcodeEvalClient(
	client func(method string, respSample interface{}) (endpoint.Endpoint, error),
) middleware.Service {
	end, _ := client("POST", ads.AdsQrcodeResp{})
	return ads.EvalAdsQrcodeEndpoints{
		EvalAdsQrcodeEP: func(ctx context.Context, req0 interface{}) (interface{}, error) {
			req1, _ := req0.(ads.SimpleReq)
			var req2 Req
			req2.Params.Limit = req1.Params.Limit
			req2.Data.URI = string(req1.Data.IMG.URI)
			return end(ctx, req2)
		},
	}
}

func AdsDetectEvalClient(
	client func(method string, respSample interface{}) (endpoint.Endpoint, error),
) middleware.Service {
	end, _ := client("POST", ads.AdsDetectionResp{})
	return ads.EvalAdsDetectEndpoints{
		EvalAdsDetectEP: func(ctx context.Context, req0 interface{}) (interface{}, error) {
			req1, _ := req0.(ads.SimpleReq)
			var req2 Req
			req2.Params.Limit = req1.Params.Limit
			req2.Data.URI = string(req1.Data.IMG.URI)
			return end(ctx, req2)
		},
	}
}

type EvalRecognitionReq struct {
	Data struct {
		URI       string `json:"uri"`
		Attribute struct {
			Detections []ads.AdsDetection `json:"detections"`
		} `json:"attribute,omitempty"`
	} `json:"data"`
}

func AdsRecognitionEvalClient(
	client func(method string, respSample interface{}) (endpoint.Endpoint, error),
) middleware.Service {
	end, _ := client("POST", ads.AdsRecognitionResp{})
	return ads.EvalAdsRecognitionEndpoints{
		EvalAdsRecognitionEP: func(ctx context.Context, req0 interface{}) (interface{}, error) {
			req1, _ := req0.(ads.AdsRecognitionReq)
			var req2 EvalRecognitionReq
			req2.Data.URI = string(req1.Data.IMG.URI)
			req2.Data.Attribute.Detections = req1.Data.Attribute.Detections
			return end(ctx, req2)
		},
	}
}

func AdsClassifierEvalClient(
	client func(method string, respSample interface{}) (endpoint.Endpoint, error),
) middleware.Service {
	end, _ := client("POST", ads.AdsClassifierResp{})
	return ads.EvalAdsClassifierEndpoints{
		EvalAdsClassifierEP: end,
	}
}

func AddEvalsDeployMode() {
	if Set != nil {
		Set.AddEvalsDeployModeOnGPU("", DeployMode2GPUCard)
		Set.AddEvalsDeployModeOnGPU("", DeployMode4GPUCard)
	}
}
