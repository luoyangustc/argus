package video

import (
	"context"
	"encoding/json"
	"sync"

	"github.com/go-kit/kit/endpoint"

	"github.com/imdario/mergo"

	"qiniu.com/argus/service/middleware"
	"qiniu.com/argus/service/scenario/biz"
	scenario "qiniu.com/argus/service/scenario/video"
	iads "qiniu.com/argus/service/service/image/ads"
	pimagesync "qiniu.com/argus/service/service/image/ads/image_sync"
	"qiniu.com/argus/service/service/video/censor/ads"
	censor "qiniu.com/argus/service/service/video/censor/video"
)

const (
	VERSION = "1.0.0"
)

func Import(serviceID string) func(interface{}) {
	return func(s0 interface{}) {
		s := s0.(scenario.VideoService)
		Init(s, serviceID)
	}
}

type Config iads.Config

func (c *Config) UnmarshalJSON(b []byte) error {
	type C Config
	c2 := C{}
	if err := json.Unmarshal(b, &c2); err != nil {
		return err
	}
	c1 := C(iads.DEFAULT)
	_ = mergo.Merge(&c2, &c1)
	*c = Config(c2)
	return nil
}

func Init(s scenario.VideoService, serviceID string) {
	var set = censor.GetSet(s, "qiniu.com/argus/service/service/video/censor/video")
	var config = Config(iads.DEFAULT)

	var (
		ss      ads.AdsOP
		once    sync.Once
		aqcSet  biz.ServiceEvalSetter
		adSet   biz.ServiceEvalSetter
		arSet   biz.ServiceEvalSetter
		acSet   biz.ServiceEvalSetter
		evalSet scenario.OPEvalSetter
	)

	newSS := func() {
		ss1, _ := aqcSet.Gen()
		aqcsrv := ss1.(iads.EvalAdsQrcodeService)

		ss2, _ := adSet.Gen()
		adsrv := ss2.(iads.EvalAdsDetectService)

		ss3, _ := arSet.Gen()
		arsrv := ss3.(iads.EvalAdsRecognitionService)

		ss4, _ := acSet.Gen()
		acsrv := ss4.(iads.EvalAdsClassifierService)

		s, _ := iads.NewAdsService(iads.Config(config), aqcsrv, adsrv, arsrv, acsrv)
		ss = ads.NewAdsOP(s)
	}

	opSet := set.RegisterOP(scenario.ServiceInfo{ID: serviceID, Version: VERSION},
		"ads", nil, func() interface{} {
			return ads.NewOP(evalSet.Gen)
		})
	evalSet = opSet.RegisterEval(func() endpoint.Endpoint {
		once.Do(newSS)
		return ss.NewEval()
	})

	aqcSet = opSet.NewEval(
		biz.ServiceEvalInfo{Name: pimagesync.EVAL_ADS_QRCODE_NAME, Version: "1.0.0"},
		pimagesync.AdsQrcodeEvalClient,
		func() middleware.ServiceEndpoints { return iads.EvalAdsQrcodeEndpoints{} },
	).SetModel(iads.EVAL_ADS_QRCODE_CONFIG).GenId()

	adSet = opSet.NewEval(
		biz.ServiceEvalInfo{Name: pimagesync.EVAL_ADS_DETECT_NAME, Version: "1.0.0"},
		pimagesync.AdsDetectEvalClient,
		func() middleware.ServiceEndpoints { return iads.EvalAdsDetectEndpoints{} },
	).SetModel(iads.EVAL_ADS_DET_CONFIG).GenId()

	arSet = opSet.NewEval(
		biz.ServiceEvalInfo{Name: pimagesync.EVAL_ADS_RECOGNITION_NAME, Version: "1.0.0"},
		pimagesync.AdsRecognitionEvalClient,
		func() middleware.ServiceEndpoints { return iads.EvalAdsRecognitionEndpoints{} },
	).SetModel(iads.EVAL_ADS_RECOGNITION_CONFIG).GenId()

	acSet = opSet.NewEval(
		biz.ServiceEvalInfo{Name: pimagesync.EVAL_ADS_CLASSIFIER_NAME, Version: "1.0.0"},
		pimagesync.AdsClassifierEvalClient,
		func() middleware.ServiceEndpoints { return iads.EvalAdsClassifierEndpoints{} },
	).SetModel(iads.EVAL_ADS_CLASSIFIER_CONFIG).GenId()

	_ = opSet.GetConfig(context.Background(), &config)
}
