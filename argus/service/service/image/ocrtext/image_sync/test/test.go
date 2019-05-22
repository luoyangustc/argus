package test

import (
	"fmt"

	// "io/ioutil"
	"bufio"
	"bytes"
	"encoding/json"
	"strconv"
	"strings"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	"qiniu.com/argus/test/biz/assert"
	P "qiniu.com/argus/test/biz/batch"
	"qiniu.com/argus/test/biz/env"
	E "qiniu.com/argus/test/biz/env"
	"qiniu.com/argus/test/biz/proto"
	T "qiniu.com/argus/test/biz/tsv"
	biz "qiniu.com/argus/test/biz/util"
	"qiniu.com/argus/test/configs"
)

var _ = Describe("OCR文本识别|/v1/ocr/text", func() {
	var server = "ocrtext"
	var path = biz.GetPath(server, "image", "path")
	var client = E.Env.GetClientArgus()
	Describe("正向用例", func() {
		Context("tsv文件验证ocr/text", func() {
			tsvFile := configs.StubConfigs.Servers.Type["image"][server].Tsv
			imageSet := configs.StubConfigs.Servers.Type["image"][server].Set

			tsv := T.NewTsv(tsvFile, imageSet, path,
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
				By("tsv文件结果:\n" + line)
				// 调用api获取最新结果
				uri := env.Env.GetURIPrivate(tsv.Set + record[0])
				By("请求地址及路径是否正确")
				// 私有化检查
				err := biz.StoreUri(tsv.Set+record[0], uri)
				if err != nil {
					panic(err)
				}
				It("测试图片:"+imageSet+record[0], func() {
					By("检查api请求: 检查服务是否正常")

					fmt.Println(record)
					fmt.Println("******************************************************")
					resp, err := client.PostWithJson(path, proto.NewArgusReq(uri, nil, nil))

					Expect(err).Should(BeNil())
					By("检验api返回http code")
					if resp.Status() == 200 {
						Expect(resp.Status()).To(Equal(200))
						var result proto.ArgOcrTextResult
						err := json.Unmarshal([]byte(record[1]), &result)
						By("检查tsv解析:\n" + line)
						Expect(err).Should(BeNil())

						// 以下为验证结果
						var resObj proto.ArgOcrTextRes
						err = json.Unmarshal(resp.BodyToByte(), &resObj)
						By("检查api返回解析:\n")
						Expect(err).Should(BeNil())

						By("检验type")
						Expect(resObj.Result.Type).To(Equal(result.Type))
						Expect(resObj.Result.Type).To(Equal(result.Type))
						for i, text := range resObj.Result.Texts {
							By("检验Texts:" + strconv.Itoa(i))
							Expect(text).To(Equal(result.Texts[i]))
						}
						// 有1个pix不匹配，经开发交流，对bboxes容忍浮动1
						for i, bboxes := range resObj.Result.Bboxes {
							By("检验Bboxes:" + strconv.Itoa(i))
							for j, bbox := range bboxes {
								for k, v := range bbox {
									Expect(float64(v)).To(BeNumerically("~", float64(result.Bboxes[i][j][k]), 1.0))
								}
							}
						}
					} else {
						T.CheckError(resp.BodyToByte(), []byte(record[1]))
					}
				})
			}

		})
	})
	Describe("反向用例", func() {
		fileName := "serving/weixinweibo-ocr/set20180112/normal_11215.jpg"
		XIt("测试图片超过大小:"+fileName, func() {
			By("测试图片：" + fileName)
			uri := env.Env.GetURIPrivate(fileName)
			resp, err := client.PostWithJson(path, proto.NewArgusReq(uri, nil, nil))

			By("检查api请求:")
			Expect(err).Should(BeNil())
			By("检验api返回http code")
			Expect(resp.Status()).To(Equal(400))

			// 以下为验证结果
			var resObj proto.ErrMessage
			err = json.Unmarshal(resp.BodyToByte(), &resObj)
			By("检查api返回解析:\n")
			Expect(err).Should(BeNil())
		})
		XIt("异常图片流测试", func() {
			imgList, err := P.BatchGetImg("normal")
			Expect(err).To(BeNil())
			for _, img := range imgList {
				By("测试图片：" + img.Imgname)
				Expect(biz.StoreUri(img.Imgname, img.Imgurl)).Should(BeNil())
				resp, err := client.PostWithJson(path,
					proto.NewArgusReq(img.Imgurl, nil, proto.NewLimit(0)))
				By("检查api请求:")
				Expect(err).Should(BeNil())

				// 以下为验证结果
				if resp.Status() == 200 {
					var resObj proto.ArgOcrTextRes
					err = json.Unmarshal(resp.BodyToByte(), &resObj)
					By("检查api返回解析:\n")
					Expect(err).Should(BeNil())
				} else if resp.Status() == 400 {
					var resObj proto.ErrMessage
					err = json.Unmarshal(resp.BodyToByte(), &resObj)
					By("检查api返回解析:\n")
					Expect(err).Should(BeNil())
				}
			}
		})
		Describe("反向用例", func() {
			Context("Check Code", func() {
				assert.CheckImageCode(client, path, nil)
			})
		})
	})
})
