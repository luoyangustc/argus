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

func EvalAdsDetection(EvalServer string, Server string, SceneType string) {
	var (
		client = E.Env.GetClientServing()
		path   = biz.GetPath(Server, SceneType, "evalpath")
		tsv    = T.NewTsv(configs.StubConfigs.Servers.Type["eval"][EvalServer].Tsv,
			configs.StubConfigs.Servers.Type["eval"][EvalServer].Set, path, configs.StubConfigs.Servers.Type["eval"][EvalServer].Precision,
			proto.NewLimit(configs.StubConfigs.Servers.Type["eval"][EvalServer].Limit))
	)
	Describe("ads-detection tsv", func() {
		Context("ads-detection tsv", func() {
			T.TsvTest(client, tsv, T.CheckCommon)
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
					By("请求eval AdsRecognition...")
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
