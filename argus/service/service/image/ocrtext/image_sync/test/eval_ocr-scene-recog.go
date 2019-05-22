package test

import (
	"bufio"
	"bytes"
	"encoding/json"
	"strconv"
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

var _ = Describe("通用场景OCR图片文字识别|/v1/eval/ocr-scene-recog", func() {
	var server = "text_ocrscenerecog"
	var pathDetect = biz.GetPath("text_ocrscenedetect", "image", "evalpath")
	var pathRecog string = biz.GetPath(server, "image", "evalpath")

	Describe("正向用例", func() {
		Context("tsv文件验证ocr-scene-recog", func() {
			var client = E.Env.GetClientServing()
			tsvFile := configs.StubConfigs.Servers.Type["image"][server].Tsv
			imageSet := configs.StubConfigs.Servers.Type["image"][server].Set

			tsv := T.NewTsv(tsvFile, imageSet, pathDetect,
				configs.StubConfigs.Servers.Type["image"][server].Precision, nil)
			buf, err := env.Env.GetTSV(tsv.Name)
			if err != nil {
				By("tsv下载失败：" + tsvFile)
				panic(err)
			}

			reader := bufio.NewReader(bytes.NewReader(buf))
			scanner := bufio.NewScanner(reader)
			scanner.Split(bufio.ScanLines)

			for scanner.Scan() {
				line := scanner.Text()
				record := strings.Split(line, "\t")
				uri := env.Env.GetURIPrivate(tsv.Set + record[0])
				// 私有化检查
				err = biz.StoreUri(tsv.Set+record[0], uri)
				if err != nil {
					panic(err)
				}
				It("测试图片:"+record[0], func() {
					// 调用api获取最新结果
					By("测试图片：" + imageSet + record[0])
					By("tsv文件结果:\n" + line)
					// 通过detect接口获取bboxes
					By("图片请求Detect Api:")
					resp, err := client.PostWithJson(pathDetect, proto.NewArgusReq(uri, nil, nil))
					By("检查detect api请求: 检查服务是否正常，请求地址及路径是否正确")
					Expect(err).Should(BeNil())
					if resp.Status() == 400 {
						var result proto.ErrMessage
						err := json.Unmarshal([]byte(record[1]), &result)
						By("检查tsv解析:\n" + line)
						Expect(err).Should(BeNil())

						var resObj proto.ErrMessage
						err = json.Unmarshal(resp.BodyToByte(), &resObj)
						By("检查detect api返回解析:\n")
						Expect(err).Should(BeNil())

						Expect(len(resObj.Message)).To(BeNumerically(">", 0))
						Expect(len(result.Message)).To(BeNumerically(">", 0))
					} else {
						var result proto.OcrSenceRecogResult
						err := json.Unmarshal([]byte(record[1]), &result)
						By("检查tsv解析:\n" + line)
						Expect(err).Should(BeNil())

						var detectRes proto.OcrSenceDetectRes
						err = json.Unmarshal(resp.BodyToByte(), &detectRes)
						By("检查detect api返回解析:\n")
						Expect(err).Should(BeNil())

						// 请求recogize
						By("图片请求Recognize Api:")
						resp, err = client.PostWithJson(pathRecog, proto.NewArgusReq(uri, nil, detectRes.Result))

						By("检查recognize api请求: 检查服务是否正常，请求地址及路径是否正确")
						Expect(err).Should(BeNil())

						var recogRes proto.OcrSenceRecogDataRes
						err = json.Unmarshal(resp.BodyToByte(), &recogRes)
						By("检查recognize api返回解析:\n")
						Expect(err).Should(BeNil())

						By("检验code")
						Expect(recogRes.Code).To(Equal(0))

						By("检验result")
						for ii, text := range recogRes.Result.Texts {
							for jj, point := range text.Bboxes {
								By("检验text:" + strconv.Itoa(ii) + ".point:" + strconv.Itoa(jj))
								Expect(float64(point)).To(BeNumerically("~", float64(result.Texts[ii].Bboxes[jj]), 1.0))
							}
							By("检验i:" + strconv.Itoa(ii) + "text")
							Expect(text.Text).To(Equal(result.Texts[ii].Text))
						}
					}
				})
			}
		})
	})
})
