package test

import (
	"bufio"
	"bytes"
	"encoding/json"
	"strings"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	"qiniu.com/argus/test/biz/env"
	E "qiniu.com/argus/test/biz/env"
	"qiniu.com/argus/test/biz/proto"
	T "qiniu.com/argus/test/biz/tsv"
	biz "qiniu.com/argus/test/biz/util"
	"qiniu.com/argus/test/configs"
)

var _ = Describe("身份证专用识别模型(高研院)", func() {

	var server = "idcard_ocrsaricrann"
	var path = biz.GetPath(server, "image", "evalpath")
	tsv := T.NewTsv(configs.StubConfigs.Servers.Type["image"][server].Tsv, configs.StubConfigs.Servers.Type["image"][server].Set, path,
		configs.StubConfigs.Servers.Type["image"][server].Precision, nil)

	Describe("正向用例", func() {
		Context("tsv文件验证", func() {
			var client = E.Env.GetClientServing()
			buf, err := env.Env.GetTSV(tsv.Name)
			if err != nil {
				By("tsv下载失败：")
				panic(err)
			}

			// test code
			// buf, err := ioutil.ReadFile("ocr-sari-crann.tsv")

			reader := bufio.NewReader(bytes.NewReader(buf))
			scanner := bufio.NewScanner(reader)
			scanner.Split(bufio.ScanLines)

			for scanner.Scan() {
				line := scanner.Text()
				record := strings.Split(line, "\t")
				uri := env.Env.GetURIPrivate(tsv.Set + record[0])
				By("检查是否支持私有化本地下载")
				err := biz.StoreUri(tsv.Set+record[0], uri)
				if err != nil {
					panic(err)
				}
				By("tsv文件结果:\n" + line)
				It("测试图片", func() {
					//调用api获取最新结果
					By("测试图片：" + tsv.Set + record[0])

					By("检验tsv解析:\n")
					var req proto.OcrSariCranParams
					if err := json.Unmarshal([]byte(record[2]), &req); err != nil {
						By("tsv解析record[2]出错，请检查此行格式:\n" + record[2])
						Expect(err).Should(BeNil())
					}
					var result proto.OcrSariCranResult
					if err = json.Unmarshal([]byte(record[3]), &result); err != nil {
						By("tsv解析record[3]出错，请检查此行格式:\n" + record[3])
						Expect(err).Should(BeNil())
					}

					// 请求api
					params := proto.NewOcrSariCrannParams(req.Bboxes)
					resp, err := client.PostWithJson(path, proto.NewArgusReq(uri, nil, params))

					By("检验api返回http status")
					Expect(resp.Status()).To(Equal(200))
					By("检查api请求: 请检查服务是否正常，请求地址及路径是否正确")
					Expect(err).Should(BeNil())

					var res proto.OcrSariCranRes
					if err = json.Unmarshal(resp.BodyToByte(), &res); err != nil {
						By("api返回解析失败，请检查接口是否有变化")
						Expect(err).Should(BeNil())
					}

					// 以下验证
					T.CheckCommon(resp.BodyToByte(), []byte(record[2]), tsv.Precision)
				})
			}
		})
	})
})
