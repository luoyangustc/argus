package test

import (
	"bufio"
	"bytes"
	"encoding/json"
	"strings"

	. "github.com/onsi/ginkgo"
	O "github.com/onsi/gomega"

	E "qiniu.com/argus/test/biz/env"
	"qiniu.com/argus/test/biz/proto"
	T "qiniu.com/argus/test/biz/tsv"
	biz "qiniu.com/argus/test/biz/util"
	"qiniu.com/argus/test/configs"
)

var _ = Describe("长文本OCR图片文字检测——CTPN模型|/v1/eval/ocr.ocr-ctpn", func() {
	var server = "bankcard_ocrctpn"
	var path = biz.GetPath(server, "image", "evalpath")
	tsv := T.NewTsv(configs.StubConfigs.Servers.Type["image"][server].Tsv, configs.StubConfigs.Servers.Type["image"][server].Set, path,
		configs.StubConfigs.Servers.Type["image"][server].Precision, nil)

	Describe("正向用例", func() {
		Context("tsv文件验证ocr-ctpn", func() {
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
				err = biz.StoreUri(tsv.Set+record[0], uri)
				if err != nil {
					panic(err)
				}
				It("测试图片:"+tsv.Set+record[0], func() {
					// 调用api获取最新结果
					By("tsv文件结果:\n" + line)
					By("请求ocr-ctpn Api:")

					resp, err := client.PostWithJson(tsv.Path, proto.NewArgusReq(uri, nil, tsv.Params))
					O.Expect(err).Should(O.BeNil())
					if resp.Status() == 200 {
						var result OcrCtpnResult
						if err := json.Unmarshal([]byte(record[1]), &result); err != nil {
							O.Expect(err).Should(O.BeNil())
						}
						T.CheckCommon(resp.BodyToByte(), []byte(record[1]), tsv.Precision)
					} else {
						T.CheckError(resp.BodyToByte(), []byte(record[1]))
					}
				})
			}
		})
	})
})

type OcrCtpnResult struct {
	Bboxes [][4][2]int `json:"bboxes"`
}
