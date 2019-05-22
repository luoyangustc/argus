package test

import (
	. "github.com/onsi/ginkgo"

	EvalTest "qiniu.com/argus/service/service/image/ads/image_sync/test/evaltest"
)

var _ = Describe("[eval]|[AdsClassifier]|/v1/eval/...ads.image_sync.evalAdsClassifier", func() {
	EvalTest.EvalAdsClassifier("AdsClassifier", "evalAdsClassifier", "image")
})

var _ = Describe("[eval]|[AdsWQrcode]|v1/eval/...ads.image_sync.evalAdsQrcode", func() {
	EvalTest.EvalAdsQrcode("AdsQrcode", "evalAdsQrcode", "image")
})

var _ = Describe("[eval|[AdsRecognition]|v1/eval/...ads.image_sync.evalAdsRecognition", func() {
	EvalTest.EvalAdsRecognition("AdsRecognition", "evalAdsRecognition", "image")
})

var _ = Describe("[eval]|[AdsDetection]|v1/eval/...ads.image_sync.evalAdsDetection", func() {
	EvalTest.EvalAdsDetection("AdsDetection", "evalAdsDetection", "image")
})
