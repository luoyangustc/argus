package test

import (
	"bufio"
	"bytes"
	"encoding/base64"
	"encoding/json"
	"strings"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	E "qiniu.com/argus/test/biz/env"
	"qiniu.com/argus/test/biz/proto"
	T "qiniu.com/argus/test/biz/tsv"
	biz "qiniu.com/argus/test/biz/util"
	"qiniu.com/argus/test/configs"
)

var _ = Describe("身份证专用模型检测、识别预处理及信息机构化(高研院)|/v1/eval/ocr-sari-id-pre", func() {
	var server = "idcard_ocrsariidpre"
	var path = biz.GetPath(server, "image", "evalpath")
	Describe("正向用例", func() {
		Context("predetect验证", func() {
			tsv := T.NewTsv(configs.StubConfigs.Servers.Type["image"][server].Tsv, configs.StubConfigs.Servers.Type["image"][server].Set, path,
				configs.StubConfigs.Servers.Type["image"][server].Precision, nil)

			var client = E.Env.GetClientServing()
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
				cutURI := E.Env.GetURIPrivateV2("ava-ocr-sari-idcard/alignedimg/20180629/cut_" + record[0])
				By("检查是否支持私有化本地下载")
				cutURIName := "ava-ocr-sari-idcard/alignedimg/20180629/cut_" + record[0]
				err = biz.StoreUri("ava-ocr-sari-idcard/alignedimg/20180629/cut_"+record[0], cutURI)
				if err != nil {
					panic(err)
				}
				By("tsv文件结果:\n" + line)
				It("测试图片", func() {
					//调用api获取最新结果
					By("测试图片：" + tsv.Set + record[0])

					By("检验tsv解析:\n")
					var result1 proto.OcrSariIDPreParams2
					if err = json.Unmarshal([]byte(record[2]), &result1); err != nil {
						By("tsv解析record[2]出错，请检查格式:\n" + record[2])
						Expect(err).Should(BeNil())
					}
					var result2 proto.OcrSariIDPreParams2
					if err = json.Unmarshal([]byte(record[3]), &result2); err != nil {
						By("tsv解析record[3]出错，请检查格式:\n" + record[3])
						Expect(err).Should(BeNil())
					}

					// 请求api
					params := proto.NewOcrSariIDPredetectParams(result1.Type)

					resp, err := client.PostWithJson(path, proto.NewArgusReq(uri, nil, params))

					By("检验api返回http status")
					Expect(resp.Status()).To(Equal(200))
					By("检查api请求: 请检查服务是否正常，请求地址及路径是否正确")
					Expect(err).Should(BeNil())

					var res proto.OcrSariIDPreRes
					if err = json.Unmarshal(resp.BodyToByte(), &res); err != nil {
						By("api返回解析失败，请检查接口是否有变化")
						Expect(err).Should(BeNil())
					}

					// 将图片转base64

					buf, err = biz.GetImgBuf(cutURIName)
					Expect(err).Should(BeNil())
					bufstore := make([]byte, len(res.Result.AlignedImg))
					base64.StdEncoding.Encode(bufstore, buf)

					// 以下验证，比较AlignedImg字段，需单独比较
					Expect(res.Result.AlignedImg).To(Equal(string(bufstore[:])))

					// 将AlignedImg设置为空，比较 Bboxes Names、Regions
					res.Result.AlignedImg = ""
					result2.AlignedImg = ""
					act, err := json.Marshal(res)
					Expect(err).Should(BeNil())
					exp, err := json.Marshal(result2)
					Expect(err).Should(BeNil())
					T.CheckCommon(act, exp, tsv.Precision)
				})
			}
		})
		Context("prerecog验证", func() {
			tsv := T.NewTsv(configs.StubConfigs.Servers.Type["image"][server].Tsvs[0], configs.StubConfigs.Servers.Type["image"][server].Sets[0], path,
				configs.StubConfigs.Servers.Type["image"][server].Precision, nil)

			var client = E.Env.GetClientServing()
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
				By("tsv文件结果:\n" + line)
				It("测试图片", func() {
					//调用api获取最新结果
					By("测试图片：" + tsv.Set + record[0])

					By("检验tsv解析:\n")
					var result1 proto.OcrSariIDPreParams
					if err = json.Unmarshal([]byte(record[2]), &result1); err != nil {
						By("tsv解析record[2]出错，请检查格式:\n" + record[2])
						Expect(err).Should(BeNil())
					}
					var result2 proto.OcrSariIDPreParams
					if err = json.Unmarshal([]byte(record[3]), &result2); err != nil {
						By("tsv解析record[3]出错，请检查格式:\n" + record[3])
						Expect(err).Should(BeNil())
					}

					// 请求api

					resp, err := client.PostWithJson(path, proto.NewArgusReq(uri, nil, result1))

					By("检验api返回http status")
					Expect(resp.Status()).To(Equal(200))
					By("检查api请求: 请检查服务是否正常，请求地址及路径是否正确")
					Expect(err).Should(BeNil())

					var res proto.OcrSariIDPreRes
					if err = json.Unmarshal(resp.BodyToByte(), &res); err != nil {
						By("api返回解析失败，请检查接口是否有变化")
						Expect(err).Should(BeNil())
					}

					// 以下验证
					T.CheckCommon(resp.BodyToByte(), []byte(record[3]), tsv.Precision)
				})
			}
		})
		Context("postprocess验证", func() {
			tsv := T.NewTsv(configs.StubConfigs.Servers.Type["image"][server].Tsvs[1], configs.StubConfigs.Servers.Type["image"][server].Sets[1], path,
				configs.StubConfigs.Servers.Type["image"][server].Precision, nil)

			var client = E.Env.GetClientServing()
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
				By("tsv文件结果:\n" + line)
				It("测试图片", func() {
					// 调用api获取最新结果
					By("测试图片：" + tsv.Set + record[0])

					By("检验tsv解析:\n")
					var result1 proto.OcrSariIDPreParams
					if err = json.Unmarshal([]byte(record[2]), &result1); err != nil {
						By("tsv解析record[2]出错，请检查格式:\n" + record[2])
						Expect(err).Should(BeNil())
					}
					var result2 proto.OcrSariIDPreParams
					if err = json.Unmarshal([]byte(record[3]), &result2); err != nil {
						By("tsv解析record[3]出错，请检查格式:\n" + record[3])
						Expect(err).Should(BeNil())
					}

					// 请求api

					resp, err := client.PostWithJson(path, proto.NewArgusReq(uri, nil, result1))

					By("检验api返回http status")
					Expect(resp.Status()).To(Equal(200))
					By("检查api请求: 请检查服务是否正常，请求地址及路径是否正确")
					Expect(err).Should(BeNil())

					var res proto.OcrSariIDPreRes
					if err = json.Unmarshal(resp.BodyToByte(), &res); err != nil {
						By("api返回解析失败，请检查接口是否有变化")
						Expect(err).Should(BeNil())
					}

					// 以下验证
					T.CheckCommon(resp.BodyToByte(), []byte(record[3]), tsv.Precision)
				})
			}
		})
	})
})
