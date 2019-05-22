package tsv

import (
	"encoding/json"

	GK "github.com/onsi/ginkgo"
	GM "github.com/onsi/gomega"

	"qiniu.com/argus/test/biz/client"
	"qiniu.com/argus/test/biz/proto"
	"qiniu.com/argus/test/lib/qnhttp"
)

// checkUnmarshalRes ... check Unmarshal response body
func checkUnmarshalRes(resp *qnhttp.Response, apiName string) {
	warning := "api返回解析失败，请检查接口是否有变化"
	switch apiName {
	case "Detect API":
		var res proto.OcrDetectRes
		if err := json.Unmarshal(resp.BodyToByte(), &res); err != nil {
			GK.By(warning)
			GM.Expect(err).Should(GM.BeNil())
		}
	case "Recognize API":
		var res proto.OcrRecognizeRes
		if err := json.Unmarshal(resp.BodyToByte(), &res); err != nil {
			GK.By(warning)
			GM.Expect(err).Should(GM.BeNil())
		}
	default:
		return
	}
}

// CheckPostAPI ... post with json API
func CheckPostAPI(uri string, apiPath string, apiName string, argusReq proto.ArgusReq) (res *qnhttp.Response, err error) {
	var client client.Client

	GK.By("图片请求" + apiName + ":")
	resp, err := client.PostWithJson(apiPath, argusReq)

	GK.By("检查" + apiName + "请求: 请检查服务是否正常，请求地址及路径是否正确")
	GM.Expect(err).Should(GM.BeNil())

	GK.By("检查api返回http code:")
	GM.Expect(resp.Status()).To(GM.Equal(200))

	checkUnmarshalRes(resp, apiName)

	return resp, err
}
