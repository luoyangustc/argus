package evaltest

import (
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	P "qiniu.com/argus/test/biz/batch"
	E "qiniu.com/argus/test/biz/env"
	"qiniu.com/argus/test/biz/proto"
	T "qiniu.com/argus/test/biz/tsv"
	biz "qiniu.com/argus/test/biz/util"
	"qiniu.com/argus/test/configs"
)

func EvalFacexFeatureTest(EvalServer string, Server string, SceneType string) {
	var path = biz.GetPath(Server, SceneType, "evalpath")
	var client = E.Env.GetClientServing()

	Describe("facex-feature-v4 tsv", func() {
		var tsv = T.NewTsv(configs.StubConfigs.Servers.Type["eval"][EvalServer].Tsv,
			configs.StubConfigs.Servers.Type["eval"][EvalServer].Set, path, configs.StubConfigs.Servers.Type["eval"][EvalServer].Precision,
			nil)
		// "serving/facex-feature/face-feature-r100-ms1m-0221/set1/201809061008.tsv",
		// 	"serving/facex-feature/set1/", path, 0.000001, nil)
		Context("facex-feature-v4 tsv", func() {
			T.TsvTestD(client, tsv, CheckFacexFeature, proto.NewPtsAttr)
		})

	})
	Describe("facex-feature-v4 功能测试", func() {
		Context("图片>4M", func() {
			fileName := "eval/pilitician/blued_1227_01096576.jpg"
			uri := E.Env.GetURIPrivate(fileName)
			if err := biz.StoreUri(fileName, uri); err != nil {
				panic(err)
			}
			It("图片>4M", func() {

				ptsStr := `[[393, 256],[1171, 256], [1171, 1250], [393, 1250]]`
				attr, err := proto.NewPtsAttr(ptsStr)
				Expect(err).Should(BeNil())
				resp, err := client.PostWithJson(path,
					proto.NewArgusReq(uri, attr, nil))
				Expect(err).Should(BeNil())
				Expect(resp.Status()).To(Equal(200))
			})
		})
		Context("异常图片", func() {
			imgSet := "test/image/face-feature/"
			fileName := "feature-abnormal1.jpeg"
			uri := E.Env.GetURIPrivate(imgSet + fileName)
			if err := biz.StoreUri(fileName, uri); err != nil {
				panic(err)
			}
			It("异常图片", func() {
				ptsStr := `[[234,146],[348,146],[348,311],[234,311]]`
				attr, err := proto.NewPtsAttr(ptsStr)
				Expect(err).Should(BeNil())
				resp, err := client.PostWithJson(path,
					proto.NewArgusReq(uri, attr, nil))
				Expect(err).Should(BeNil())
				Expect(resp.Status()).To(Equal(200))
			})
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
				It("测试图片:"+imgName, func() {
					ptsStr := `[[234,146],[348,146],[348,311],[234,311]]`
					attr, _ := proto.NewPtsAttr(ptsStr)
					resp, err := client.PostWithJson(path,
						proto.NewArgusReq(imgUrl, attr, nil))
					Expect(err).Should(BeNil())
					Expect(resp.Status()).To(Or(Equal(200), BeNumerically("<", 600)))
					Expect(err).Should(BeNil())
				})
			}
		})
	})
}
