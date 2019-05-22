package test

import (
	"encoding/json"
	"fmt"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	P "qiniu.com/argus/test/biz/batch"
	E "qiniu.com/argus/test/biz/env"
	"qiniu.com/argus/test/biz/proto"
	T "qiniu.com/argus/test/biz/tsv"
	biz "qiniu.com/argus/test/biz/util"
	"qiniu.com/argus/test/configs"
)

func EvalTerrorDetectTest(EvalServer string, Server string, SceneType string) {
	var client = E.Env.GetClientServing()
	var path = biz.GetPath(Server, SceneType, "evalpath")
	//update 20180920 ATLAB-8269
	var tsv = T.NewTsv(configs.StubConfigs.Servers.Type["eval"][EvalServer].Tsv,
		configs.StubConfigs.Servers.Type["eval"][EvalServer].Set, path, configs.StubConfigs.Servers.Type["eval"][EvalServer].Precision, nil)
	Describe("测试暴恐检测tsv", func() {
		Context("测试暴恐检测tsv", func() {
			T.TsvTest(client, tsv, T.CheckCommon)
		})
	})
	Describe("功能测试", func() {
		image := []string{"long.png.jpeg"}
		imageSet := "test/image/"
		for _, file := range image {
			fileName := imageSet + file
			//调用api获取最新结果
			uri := E.Env.GetURIPrivate(fileName)
			err := biz.StoreUri(fileName, uri)
			if err != nil {
				panic(err)
			}
			fmt.Println("下载图片" + fileName)
			It("测试图片:"+fileName, func() {
				resp, err := client.PostWithJson(path,
					proto.NewArgusReq(uri, nil, nil))
				Expect(err).Should(BeNil())
				Expect(resp.Status()).To(Equal(400))
			})
		}
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
