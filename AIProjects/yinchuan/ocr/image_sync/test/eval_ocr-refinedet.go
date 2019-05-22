package test

import (

	// "io/ioutil"
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

var _ = Describe("锐安银川网安定制图像类型分类模型|/v1/eval/yinchuanclassify.ocr-refinedet", func() {
	var server = "ocryinchuan_ocrrefinedet"
	var path = biz.GetPath(server, "image", "evalpath")
	// string = "/v1/eval/yinchuanclassify.ocr-refinedet"
	var client = E.Env.GetClientServing()

	Describe("正向用例", func() {
		Context("tsv文件验证ocr/yinchuan", func() {
			// tsvFile := "ava-ocr-refinedet/20181019/ocr-refinedet-argus.tsv"
			// imageSet := "ava-ocr-refinedet/"
			//https://jira.qiniu.io/browse/ATLAB-9236
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
				uri := env.Env.GetURIPrivate(tsv.Set + record[0])
				// 私有化检查
				err = biz.StoreUri(tsv.Set+record[0], uri)
				if err != nil {
					By(uri + "图片获取失败")
					panic(err)
				}
				It("测试图片:"+imageSet+record[0], func() {

					By("tsv文件结果:\n" + line)

					var result []Result
					err := json.Unmarshal([]byte(record[1]), &result)
					Expect(err).Should(BeNil())

					// 调用api获取最新结果
					resp, err := client.PostWithJson(path, proto.NewArgusReq(uri, nil, nil))

					By("检查api请求:")
					Expect(err).Should(BeNil())
					By("检验api http code")
					Expect(resp.Status()).To(Equal(200))

					// 以下验证
					T.CheckInterface(resp.BodyToByte(), []byte(record[1]), tsv.Precision)
				})
			}

		})
	})
})
