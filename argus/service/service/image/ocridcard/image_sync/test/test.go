package test

import (
	// "io/ioutil"
	"bufio"
	"bytes"
	"encoding/base64"
	"encoding/json"
	"strings"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"qiniu.com/argus/test/biz/assert"
	P "qiniu.com/argus/test/biz/batch"
	E "qiniu.com/argus/test/biz/env"
	"qiniu.com/argus/test/biz/proto"
	T "qiniu.com/argus/test/biz/tsv"
	biz "qiniu.com/argus/test/biz/util"
	"qiniu.com/argus/test/configs"
)

var _ = Describe("身份证专用识别模型(高研院)|/v1/ocr/idcard", func() {
	var client = E.Env.GetClientArgus()
	var server = "ocridcard"
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
				uri := E.Env.GetURIPrivate(tsv.Set + record[0])
				By("检查是否支持私有化本地下载")
				err = biz.StoreUri(tsv.Set+record[0], uri)
				if err != nil {
					panic(err)
				}
				By("检查是否支持私有化本地下载")
				cutURI := E.Env.GetURIPrivateV2("ava-ocr-sari-idcard/alignedimg/20180629/cut_" + record[0])
				err = biz.StoreUri("ava-ocr-sari-idcard/alignedimg/20180629/cut_"+record[0], cutURI)
				if err != nil {
					panic(err)
				}
				By("tsv文件结果:\n" + line)
				It("测试图片", func() {
					// 调用api获取最新结果
					By("测试图片：" + tsv.Set + record[0])

					By("检验tsv解析:\n")
					var result proto.OcrSariIDCardResult
					if err = json.Unmarshal([]byte(record[1]), &result); err != nil {
						By("tsv解析record[1]出错，请检查此行格式:\n" + line)
						Expect(err).Should(BeNil())
					}

					// 请求api

					resp, err := client.PostWithJson(path, proto.NewArgusReq(uri, nil, tsv.Params))

					By("检验api返回http status")
					Expect(resp.Status()).To(Equal(200))
					By("检查api请求: 请检查服务是否正常，请求地址及路径是否正确")
					Expect(err).Should(BeNil())

					var res proto.OcrSariIDCardRes
					if err = json.Unmarshal(resp.BodyToByte(), &res); err != nil {
						By("api返回解析失败，请检查接口是否有变化")
						Expect(err).Should(BeNil())
					}

					// 将图片转base64

					cutURIName := "ava-ocr-sari-idcard/alignedimg/20180629/cut_" + record[0]
					buf, err = biz.GetImgBuf(cutURIName)
					Expect(err).Should(BeNil())
					bufstore := make([]byte, len(res.Result.URI))
					base64.StdEncoding.Encode(bufstore, buf)

					// 比较URI字段，需单独比较
					Expect(res.Result.URI).To(Equal(string(bufstore[:])))
					// 将URI设置为空，比较 Bboxes 、Texts字段
					res.Result.URI = ""
					result.URI = ""
					act, err := json.Marshal(res)
					Expect(err).Should(BeNil())
					exp, err := json.Marshal(result)
					Expect(err).Should(BeNil())
					T.CheckCommon(act, exp, tsv.Precision)
				})
			}
			XIt("异常图片流测试", func() {
				imgList, err := P.BatchGetImg("normal")
				Expect(err).To(BeNil())
				for _, img := range imgList {
					By("测试图片：" + img.Imgname)
					Expect(biz.StoreUri(img.Imgname, img.Imgurl)).Should(BeNil())
					resp, err := client.PostWithJson(path,
						proto.NewArgusReq(img.Imgurl, nil, proto.NewLimit(0)))
					Expect(err).Should(BeNil())
					Expect(resp.Status()).To(Or(Equal(200), BeNumerically("<", 500)))
				}
			})
		})
	})
	Describe("反向用例", func() {
		Context("Check Code", func() {
			assert.CheckImageCode(client, path, nil)
		})
	})
})
