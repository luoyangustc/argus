package test

import (
	"encoding/json"
	"fmt"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	EvalTest "qiniu.com/argus/service/service/image/pulp/image_sync/test/evaltest"
	"qiniu.com/argus/test/biz/assert"
	P "qiniu.com/argus/test/biz/batch"
	E "qiniu.com/argus/test/biz/env"
	"qiniu.com/argus/test/biz/proto"
	biz "qiniu.com/argus/test/biz/util"
)

var _ = Describe("[pulp]|[argus]|/v1/pulp", func() {
	var clientArgus = E.Env.GetClientArgus()
	var clientServing = E.Env.GetClientServing()
	var server = "pulp"
	var path = biz.GetPath(server, "image", "path")
	var limit = proto.NewLimit(3)
	Describe("剑皇串联", func() {
		Context("pulp-filter过滤", func() {
			var pathPulp string = biz.GetPath("evalPulpFilter", "image", "evalpath")
			image := []string{"Image-tupu-2016-09-01-17-40-382.jpg"}
			imageSet := "serving/pulp/set1/"
			for _, file := range image {
				fileName := imageSet + file
				By("测试图片：" + fileName)
				uri := E.Env.GetURIPrivate(fileName)
				if err := biz.StoreUri(fileName, uri); err != nil {
					panic(err)
				}
				It("测试图片:"+fileName, func() {
					//调用api获取最新结果
					resp, err := clientArgus.PostWithJson(path,
						proto.NewArgusReq(uri, nil, limit))
					Expect(err).Should(BeNil())
					Expect(resp.Status()).To(Equal(200))
					respServ, errServ := clientServing.PostWithJson(pathPulp,
						proto.NewArgusReq(E.Env.GetURIPrivate(fileName), nil, limit))
					Expect(errServ).Should(BeNil())
					Expect(respServ.Status()).To(Equal(200))
					CheckFirst(resp.BodyToByte(), respServ.BodyToByte(), 0.001)
				})
			}
		})
		Context("pulp-filter -> pulp", func() {
			var pathPulp string = biz.GetPath(server, "image", "evalpath")
			image := []string{"Image-tupu-2016-09-01-15-40-2912.jpg"}
			imageSet := "serving/pulp/set1/"
			for _, file := range image {
				fileName := imageSet + file
				By("测试图片：" + fileName)
				uri := E.Env.GetURIPrivate(fileName)
				if err := biz.StoreUri(fileName, uri); err != nil {
					panic(err)
				}
				It("测试图片:"+fileName, func() {
					//调用api获取最新结果
					resp, err := clientArgus.PostWithJson(path,
						proto.NewArgusReq(uri, nil, limit))
					Expect(err).Should(BeNil())
					Expect(resp.Status()).To(Equal(200))
					respServ, errServ := clientServing.PostWithJson(pathPulp,
						proto.NewArgusReq(E.Env.GetURIPrivate(fileName), nil, limit))
					Expect(errServ).Should(BeNil())
					Expect(respServ.Status()).To(Equal(200))
					Check(resp.BodyToByte(), respServ.BodyToByte(), 0.001)
				})
			}
		})
	})
	Describe("正向用例", func() {
		Context("大图pulp", func() {
			var pathPulp string = biz.GetPath(server, "image", "evalpath")
			images := []string{"Image-tupu-2016-09-01-15-40-2912.jpg", "Image-tupu-2016-09-01-05-00-1612.jpg", "Image-tupu-2016-09-01-11-10-5582.jpg"}
			imageSet := "serving/pulp/set1/"
			for _, file := range images {
				fileName := imageSet + file
				url := E.Env.GetURIPrivate(fileName)
				if err := biz.StoreUri(fileName, url); err != nil {
					panic(err)
				}
				It("测试图片:"+fileName, func() {
					//调用api获取最新结果
					resp, err := clientArgus.PostWithJson(path,
						proto.NewArgusReq(url, nil, limit))
					Expect(err).Should(BeNil())
					Expect(resp.Status()).To(Equal(200))
					respServ, errServ := clientServing.PostWithJson(pathPulp,
						proto.NewArgusReq(url, nil, limit))
					Expect(errServ).Should(BeNil())
					Expect(respServ.Status()).To(Equal(200))
					Check(resp.BodyToByte(), respServ.BodyToByte(), 0.001)
				})
			}
		})
		Context("小图pulp", func() {
			var pathPulp string = biz.GetPath("evalPulpFilter", "image", "evalpath")
			images := []string{"Image-tupu-2016-09-02-15-40-3134.jpg", "Image-tupu-2016-09-02-06-00-1138.jpg"}
			imageSet := "serving/pulp/set1/"
			for _, file := range images {
				fileName := imageSet + file
				url := E.Env.GetURIPrivate(fileName)
				if err := biz.StoreUri(fileName, url); err != nil {
					panic(err)
				}
				It("测试图片:"+fileName, func() {
					//调用api获取最新结果
					resp, err := clientArgus.PostWithJson(path,
						proto.NewArgusReq(url, nil, limit))
					Expect(err).Should(BeNil())
					Expect(resp.Status()).To(Equal(200))
					respServ, errServ := clientServing.PostWithJson(pathPulp,
						proto.NewArgusReq(url, nil, limit))
					Expect(errServ).Should(BeNil())
					Expect(respServ.Status()).To(Equal(200))
					fmt.Println(string(respServ.BodyToByte()))
					CheckFirst(resp.BodyToByte(), respServ.BodyToByte(), 0.001)
				})
			}
		})
		// 大图小图新逻辑，只走大图pulp
		Context("大图小图pulp", func() {
			var pathPulp string = biz.GetPath("evalPulpFilter", "image", "evalpath")
			image := []string{"Image-tupu-2016-09-02-16-00-3045.jpg"}
			imageSet := "serving/pulp/set1/"
			for _, file := range image {
				fileName := imageSet + file
				uri := E.Env.GetURIPrivate(fileName)
				if err := biz.StoreUri(fileName, uri); err != nil {
					panic(err)
				}
				It("测试图片:"+fileName, func() {
					//调用api获取最新结果
					resp, err := clientArgus.PostWithJson(path,
						proto.NewArgusReq(uri, nil, limit))
					Expect(err).Should(BeNil())
					Expect(resp.Status()).To(Equal(200))
					respServ, errServ := clientServing.PostWithJson(pathPulp,
						proto.NewArgusReq(E.Env.GetURIPrivate(fileName), nil, limit))
					Expect(errServ).Should(BeNil())
					Expect(respServ.Status()).To(Equal(200))
					Check(resp.BodyToByte(), respServ.BodyToByte(), 0.001)
				})
			}

		})

		Context("功能验证", func() {
			//功能性验证
			pulpfiles := []string{"test-pulp7.png", "test-pulp7.bmp",
				"pulpnormal.jpg", "pulppulp.jpg", "pulpsexy.jpg", "sexytest.jpg",
				"upload.gif", "sexy001.gif", "sexy002.gif",
				"pulpnormal.webp", "pulpsexy.webp", "pulppulp.webp"}
			for _, file := range pulpfiles {
				fileName := file
				uri := E.Env.GetURIPrivate(fileName)
				if err := biz.StoreUri(fileName, uri); err != nil {
					panic(err)
				}
				It("arguspulp－验证功能", func() {
					//调用api获取最新结果
					resp, err := clientArgus.PostWithJson(path,
						proto.NewArgusReq(uri, nil, limit))
					Expect(err).Should(BeNil())
					Expect(resp.Status()).To(Equal(200))
					//以下为验证结果
					var resObj proto.ArgusRes
					Expect(json.Unmarshal(resp.BodyToByte(), &resObj)).Should(BeNil())
					if resObj.Result.Score < 0.6 {
						Expect(resObj.Result.Review).To(Equal(true))
					} else {
						Expect(resObj.Result.Review).To(Equal(false))
					}
				})
			}
		})
		Context("limit:1", func() {
			var pathPulp string = biz.GetPath(server, "image", "evalpath")
			images := []string{"Image-tupu-2016-09-01-15-40-2912.jpg", "Image-tupu-2016-09-01-05-00-1612.jpg", "Image-tupu-2016-09-01-11-10-5582.jpg"}
			imageSet := "serving/pulp/set1/"
			var limit = proto.NewLimit(1)
			for _, file := range images {
				fileName := imageSet + file
				url := E.Env.GetURIPrivate(fileName)
				if err := biz.StoreUri(fileName, url); err != nil {
					panic(err)
				}
				It("测试图片:"+fileName, func() {
					//调用api获取最新结果
					resp, err := clientArgus.PostWithJson(path,
						proto.NewArgusReq(url, nil, limit))
					Expect(err).Should(BeNil())
					Expect(resp.Status()).To(Equal(200))
					var resObj proto.ArgusRes
					err1 := json.Unmarshal(resp.BodyToByte(), &resObj)
					Expect(err1).Should(BeNil())
					respServ, errServ := clientServing.PostWithJson(pathPulp,
						proto.NewArgusReq(E.Env.GetURIPrivate(fileName), nil, limit))
					Expect(errServ).Should(BeNil())
					Expect(respServ.Status()).To(Equal(200))
					var resObjExp proto.ArgusRes
					Expect(json.Unmarshal(respServ.BodyToByte(), &resObjExp)).Should(BeNil())
					Expect(resObj.Result.Label).Should(Equal(resObjExp.Result.Confidences[0].Index))
					Expect(resObj.Result.Score).Should(BeNumerically("~", resObjExp.Result.Confidences[0].Score, 0.001))

				})
			}
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
				if imgName == "difftype-4ch.webp" {
					continue
				}
				err = biz.StoreUri(imgName, imgUrl)
				if err != nil {
					panic(err)
				}
				It("测试图片："+imgName, func() {
					By("请求argus politician...")
					resp, err := clientArgus.PostWithJson(path,
						proto.NewArgusReq(imgUrl, nil, nil))
					Expect(err).Should(BeNil())
					Expect(resp.Status()).To(Or(Equal(200), BeNumerically("<", 500)))
					var resObj proto.ArgusRes
					err = json.Unmarshal(resp.BodyToByte(), &resObj)
					Expect(err).Should(BeNil())
				})
			}
		})
		Context("Check Code", func() {
			assert.CheckImageCode(clientArgus, path, nil)
		})
	})
})

var _ = Describe("[eval]|[Pulp]|/v1/eval/...pulp.image_sync.evalPulp", func() {
	EvalTest.EvalPulpTest("Pulp", "pulp", "image")
})

var _ = Describe("[eval]|[PulpFilter]|/v1/eval/...pulp.image_sync.evalPulpFilter", func() {
	EvalTest.EvalPulpFilterTest("PulpFilter", "evalPulpFilter", "image")
})
