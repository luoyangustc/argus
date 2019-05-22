package evaltest

import (
	"encoding/json"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	P "qiniu.com/argus/test/biz/batch"
	E "qiniu.com/argus/test/biz/env"
	"qiniu.com/argus/test/biz/proto"
	T "qiniu.com/argus/test/biz/tsv"
	biz "qiniu.com/argus/test/biz/util"
	"qiniu.com/argus/test/configs"
)

func EvalPulpFilterTest(EvalServer string, Server string, SceneType string) {
	var client = E.Env.GetClientServing()
	var path = biz.GetPath(Server, SceneType, "evalpath")
	// update:2018.09.012  https://jira.qiniu.io/browse/ATLAB-8116
	var tsv = T.NewTsv(configs.StubConfigs.Servers.Type["eval"][EvalServer].Tsv,
		configs.StubConfigs.Servers.Type["eval"][EvalServer].Set, path, configs.StubConfigs.Servers.Type["eval"][EvalServer].Precision,
		proto.NewLimit(configs.StubConfigs.Servers.Type["eval"][EvalServer].Limit))

	Describe("测试fiter剑皇tsv", func() {
		Context("测试fiter剑皇tsv", func() {
			T.TsvTest(client, tsv, T.CheckClassifyEasy)
		})
	})
	Describe("测试图片库", func() {
		Context("normal", func() {
			imgList, err := P.BatchGetImg("normal")
			if err != nil {
				panic(err)
			}
			for _, img := range imgList {
				imgName := img.Imgname
				imgUrl := img.Imgurl
				err = biz.StoreUri(imgName, imgUrl)
				if err != nil {
					panic(err)
				}
				It("测试图片："+imgName, func() {
					By("请求argus politician...")
					resp, err := client.PostWithJson(path,
						proto.NewArgusReq(imgUrl, nil, nil))
					Expect(err).Should(BeNil())
					Expect(resp.Status()).To(Or(Equal(200), BeNumerically("<", 600)))
					var resObj proto.ArgusRes
					err = json.Unmarshal(resp.BodyToByte(), &resObj)
					Expect(err).Should(BeNil())
				})
			}
		})

	})
}
