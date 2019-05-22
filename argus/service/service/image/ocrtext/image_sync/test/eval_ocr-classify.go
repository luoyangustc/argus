package test

import (
	. "github.com/onsi/ginkgo"
	E "qiniu.com/argus/test/biz/env"
	T "qiniu.com/argus/test/biz/tsv"
	biz "qiniu.com/argus/test/biz/util"
	"qiniu.com/argus/test/configs"
)

var _ = Describe("OCR图片类型分类|/v1/eval/ocrtext.ocr-classify", func() {
	var client = E.Env.GetClientServing()
	var server = "text_ocrclassify"
	var path = biz.GetPath(server, "image", "evalpath")
	tsv := T.NewTsv(configs.StubConfigs.Servers.Type["image"][server].Tsv, configs.StubConfigs.Servers.Type["image"][server].Set, path,
		configs.StubConfigs.Servers.Type["image"][server].Precision, nil)

	Describe("正向用例", func() {
		Context("tsv文件验证ocr-classify", func() {
			T.TsvTest(client, tsv, T.CheckOcrClassify)
		})
	})
	Describe("反向用例", func() {
		filepath := "serving/weixinweibo-ocr/set20180112/WechatIMG284.jpeg"
		T.CheckImageTooLarge(client, tsv, filepath)
	})
})
