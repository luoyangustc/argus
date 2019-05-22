package test

import (
	// "io/ioutil"
	"bufio"
	"bytes"
	"encoding/json"
	"strings"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	"qiniu.com/argus/test/biz/assert"
	E "qiniu.com/argus/test/biz/env"
	"qiniu.com/argus/test/biz/proto"
	T "qiniu.com/argus/test/biz/tsv"
	biz "qiniu.com/argus/test/biz/util"
	"qiniu.com/argus/test/configs"
)

var _ = Describe("银行卡识别|/v1/ocr/bankcard", func() {
	var client = E.Env.GetClientArgus()
	var server = "ocrbankcard"
	var path = biz.GetPath(server, "image", "path")
	tsv := T.NewTsv(configs.StubConfigs.Servers.Type["image"][server].Tsv, configs.StubConfigs.Servers.Type["image"][server].Set, path,
		configs.StubConfigs.Servers.Type["image"][server].Precision, nil)

	Describe("正向用例", func() {
		Context("tsv文件验证", func() {
			buf, err := E.Env.GetTSV(tsv.Name)
			if err != nil {
				By("tsv下载失败：")
				panic(err)
			}
			reader := bufio.NewReader(bytes.NewReader(buf))
			scanner := bufio.NewScanner(reader)
			scanner.Split(bufio.ScanLines)

			for scanner.Scan() {
				line := scanner.Text()
				record := strings.Split(line, "\t")

				By("检验tsv解析:\n")
				var result Result
				if tsverr := json.Unmarshal([]byte(record[1]), &result); err != nil {
					By("tsv解析record[1]出错，请检查此行格式:\n" + line)
					panic(tsverr)
				}
				uri := E.Env.GetURIPrivate(tsv.Set + record[0])
				By("检查是否支持私有化本地下载")
				err = biz.StoreUri(tsv.Set+record[0], uri)
				if err != nil {
					panic(err)
				}

				By("tsv文件结果:\n" + line)
				It("测试图片："+tsv.Set+record[0], func() {
					// 请求api
					resp, err := client.PostWithJson(path, proto.NewArgusReq(uri, nil, tsv.Params))
					By("检验api返回http status")
					Expect(resp.Status()).To(Equal(200))
					By("检查api请求: 请检查服务是否正常，请求地址及路径是否正确")
					Expect(err).Should(BeNil())
					var res Resp
					if err = json.Unmarshal(resp.BodyToByte(), &res); err != nil {
						By("api返回解析失败，请检查接口是否有变化")
						Expect(err).Should(BeNil())
					}
					// 以下验证
					T.CheckCommon(resp.BodyToByte(), []byte(record[1]), tsv.Precision)
				})
			}
		})
	})
	Describe("反向用例", func() {
		Context("Check Code", func() {
			assert.CheckImageCode(client, path, nil)
		})
	})
})

// Result ... the expect result
type Result struct {
	Bboxes [][4][2]float32        `json:"bboxes"`
	Res    map[string]interface{} `json:"res"`
}

// Resp ... the actual response
type Resp struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Result  Result `json:"result"`
}
