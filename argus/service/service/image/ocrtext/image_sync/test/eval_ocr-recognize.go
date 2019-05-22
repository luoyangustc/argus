package test

import (
	"bufio"
	"bytes"
	"encoding/json"
	"strings"

	. "github.com/onsi/ginkgo"
	O "github.com/onsi/gomega"
	"qiniu.com/argus/test/biz/env"
	E "qiniu.com/argus/test/biz/env"
	"qiniu.com/argus/test/biz/proto"
	T "qiniu.com/argus/test/biz/tsv"
	biz "qiniu.com/argus/test/biz/util"
	"qiniu.com/argus/test/configs"
)

var _ = Describe("OCR图片文字检测|/v1/eval/ocr-classify", func() {
	var server = "text_ocrrecognize"
	var pathClassify = biz.GetPath("text_ocrclassify", "image", "evalpath")
	var pathDetect = biz.GetPath("text_ocrctpn", "image", "evalpath")
	var pathRecognize = biz.GetPath(server, "image", "evalpath")
	tsv := T.NewTsv(configs.StubConfigs.Servers.Type["image"][server].Tsv, configs.StubConfigs.Servers.Type["image"][server].Set, pathClassify,
		configs.StubConfigs.Servers.Type["image"][server].Precision, nil)

	Describe("正向用例", func() {
		Context("tsv文件验证ocr-detect", func() {
			var client = E.Env.GetClientServing()
			buf, err := env.Env.GetTSV(tsv.Name)
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
				uri := env.Env.GetURIPrivate(tsv.Set + record[0])
				bizerr := biz.StoreUri(tsv.Set+record[0], uri)
				if bizerr != nil {
					panic(bizerr)
				}
				It("测试图片:"+tsv.Set+record[0], func() {
					//调用api获取最新结果
					By("测试图片：" + tsv.Set + record[0])
					By("tsv文件结果:\n" + line)
					//通过classify接口获取img_type
					By("图片请求Classify Api:")

					resp, err := client.PostWithJson(tsv.Path, proto.NewArgusReq(uri, nil, tsv.Params))

					By("检查服务是否正常，请求地址及路径是否正确:")
					O.Expect(err).Should(O.BeNil())

					if resp.Status() == 200 {
						var result proto.OcrRecognizeResult
						if err := json.Unmarshal([]byte(record[1]), &result); err != nil {
							By("tsv解析出错，请检查此行格式:\n" + line)
							O.Expect(err).Should(O.BeNil())
						}
						var classifyRes proto.ArgusRes
						if err := json.Unmarshal(resp.BodyToByte(), &classifyRes); err != nil {
							By("classify api返回解析失败，请检查接口是否有变化")
							O.Expect(err).Should(O.BeNil())
						}
						imgType := classifyRes.Result.Confidences[0].Class

						//通过detect接口获取bboxes
						By("图片请求Detect Api:")
						resp, err := client.PostWithJson(pathDetect, proto.NewArgusReq(uri, nil, tsv.Params))

						By("检查detect api请求: 请检查服务是否正常，请求地址及路径是否正确")
						O.Expect(err).Should(O.BeNil())

						var detectRes proto.OcrCtpnRes
						if err := json.Unmarshal(resp.BodyToByte(), &detectRes); err != nil {
							By("detect api返回解析失败，请检查接口是否有变化")
							O.Expect(err).Should(O.BeNil())
						}

						var ctpnBoxes = detectRes.Result.Bboxes
						var boxes [][4]int
						for i := 0; i < len(ctpnBoxes); i++ {
							boxes = append(boxes, [4]int{ctpnBoxes[i][0][0], ctpnBoxes[i][0][1], ctpnBoxes[i][2][0], ctpnBoxes[i][2][1]})
						}

						//测试recognize接口
						By("图片请求Recognize Api:")
						resp, err = client.PostWithJson(pathRecognize, proto.NewArgusReq(uri, nil, proto.NewOcrRecognizeParams(imgType, boxes)))

						By("检查recognize api请求: 请检查服务是否正常，请求地址及路径是否正确")
						O.Expect(err).Should(O.BeNil())

						By("检验api返回http code")
						O.Expect(resp.Status()).To(O.Equal(200))

						var recognizeRes proto.OcrRecognizeRes
						if err := json.Unmarshal(resp.BodyToByte(), &recognizeRes); err != nil {
							By("recognize api返回解析失败，请检查接口是否有变化")
							O.Expect(err).Should(O.BeNil())
						}
						By("检验识别结果")
						T.CheckCommon(resp.BodyToByte(), []byte(record[1]), tsv.Precision)
					} else {
						T.CheckError(resp.BodyToByte(), []byte(record[1]))
					}
				})
			}
		})
	})
})
