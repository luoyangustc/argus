package test

import (
	"encoding/json"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	Face "qiniu.com/argus/service/service/image/face"
	evaltest "qiniu.com/argus/service/service/image/face/image_sync/test/evaltest"
	assert "qiniu.com/argus/test/biz/assert"
	E "qiniu.com/argus/test/biz/env"
	"qiniu.com/argus/test/biz/proto"
	biz "qiniu.com/argus/test/biz/util"
)

// https://github.com/qbox/ava/blob/dev/docs/Argus.api.md#facecluster
var _ = Describe("[sim]|[face]|/v1/face/sim", func() {
	var client = E.Env.GetClientArgus()
	var server = "facesim"
	var path = biz.GetPath(server, "image", "path") //configs.Configs.Servers[server]["image"].Version + configs.Configs.Servers[server]["image"].Path

	Describe("正向用例", func() {
		Context("相似人脸", func() {
			var imgList = []string{"face/sim/Richard Madden_48615.jpg", "face/sim/Richard Madden_48616.jpg"}
			var uriList []string
			for _, img := range imgList {
				uri := E.Env.GetURIPrivate(img)
				err := biz.StoreUri(img, uri)
				if err != nil {
					panic(err)
				}
				uriList = append(uriList, uri)
			}
			It("测试图片", func() {
				inputData := proto.NewArgusBatchReq(uriList[0], nil, nil)
				inputData.Add(uriList[1], nil)
				By("图片请求Api:")
				resp, err := client.PostWithJson(path, inputData)
				Expect(err).Should(BeNil())
				By("检验api返回http code")
				Expect(resp.Status()).To(Equal(200))
				var faceresp Face.FaceSimResp
				Expect(json.Unmarshal(resp.BodyToByte(), &faceresp)).Should(BeNil())
				Expect(faceresp.Code).To(Equal(0))
				Expect(len(faceresp.Result.Faces)).To(Equal(2))
				for _, facebox := range faceresp.Result.Faces {
					assert.CheckPts(facebox.Pts)
					Expect(facebox.Score).To(BeNumerically(">", 0.6))
				}
				Expect(faceresp.Result.Similarity).To(BeNumerically(">", 0.6))
				Expect(faceresp.Result.Similarity).To(BeNumerically("<=", 1.0))
				Expect(faceresp.Result.Same).To(Equal(true))
			})
		})
		Context("不相似人脸", func() {
			var imgList = []string{"face/Audrey_Hepburn.jpg", "faceage.jpg"}
			var uriList []string
			for _, img := range imgList {
				uri := E.Env.GetURIPrivate(img)
				err := biz.StoreUri(img, uri)
				if err != nil {
					panic(err)
				}
				uriList = append(uriList, uri)
			}
			It("测试图片", func() {
				inputData := proto.NewArgusBatchReq(uriList[0], nil, nil)
				inputData.Add(uriList[1], nil)

				By("图片请求Api:")
				resp, err := client.PostWithJson(path, inputData)
				Expect(err).Should(BeNil())
				By("检验api返回http code")
				Expect(resp.Status()).To(Equal(200))
				var faceresp Face.FaceSimResp
				Expect(json.Unmarshal(resp.BodyToByte(), &faceresp)).Should(BeNil())
				Expect(faceresp.Code).To(Equal(0))
				Expect(len(faceresp.Result.Faces)).To(Equal(2))
				for _, face := range faceresp.Result.Faces {
					assert.CheckPts(face.Pts)
					Expect(face.Score).To(BeNumerically(">", 0.999))
				}
				Expect(faceresp.Result.Similarity).To(BeNumerically("<", 0.4))
				Expect(faceresp.Result.Same).To(Equal(false))
			})
		})
		Context("多张人脸", func() {
			var imgList = []string{"face08.jpg", "face08.jpg"}
			var uriList []string
			for _, img := range imgList {
				uri := E.Env.GetURIPrivate(img)
				err := biz.StoreUri(img, uri)
				if err != nil {
					panic(err)
				}
				uriList = append(uriList, uri)
			}
			It("测试图片", func() {
				inputData := proto.NewArgusBatchReq(uriList[0], nil, nil)
				inputData.Add(uriList[1], nil)
				By("图片请求Api:")
				resp, err := client.PostWithJson(path, inputData)
				Expect(err).Should(BeNil())
				By("检验api返回http code")
				Expect(resp.Status()).To(Equal(200))
				var faceresp Face.FaceSimResp
				Expect(json.Unmarshal(resp.BodyToByte(), &faceresp)).Should(BeNil())
				Expect(faceresp.Code).To(Equal(0))
				Expect(len(faceresp.Result.Faces)).To(Equal(2))
				for _, face := range faceresp.Result.Faces {
					assert.CheckPts(face.Pts)
					Expect(face.Score).To(BeNumerically(">", 0.999))
				}
				Expect(faceresp.Result.Similarity).To(BeNumerically(">", 0.999))
				//Expect(faceresp.Result.Similarity).To(BeNumerically("<=", 1.0))
			})
		})
		Context("非人脸", func() {
			var imgList = []string{"faceage.jpg", "CoffeeRoom.jpg"}
			var uriList []string
			for _, img := range imgList {
				uri := E.Env.GetURIPrivate(img)
				err := biz.StoreUri(img, uri)
				if err != nil {
					panic(err)
				}
				uriList = append(uriList, uri)
			}
			It("测试图片", func() {
				inputData := proto.NewArgusBatchReq(uriList[0], nil, nil)
				inputData.Add(uriList[1], nil)
				By("图片请求Api:")
				resp, err := client.PostWithJson(path, inputData)
				Expect(err).Should(BeNil())
				By("检验api返回http code")
				Expect(resp.Status()).To(Equal(400))
			})
		})
		Context("相似度负值", func() {
			var imgList = []string{"argus/face-sim/qa/face2.jpg", "argus/face-sim/qa/face1.jpg"}
			var uriList []string
			for _, img := range imgList {
				uri := E.Env.GetURIPrivate(img)
				err := biz.StoreUri(img, uri)
				if err != nil {
					panic(err)
				}
				uriList = append(uriList, uri)
			}
			It("测试图片", func() {
				inputData := proto.NewArgusBatchReq(uriList[0], nil, nil)
				inputData.Add(uriList[1], nil)
				By("图片请求Api:")
				resp, err := client.PostWithJson(path, inputData)
				Expect(err).Should(BeNil())
				By("检验api返回http code")
				Expect(resp.Status()).To(Equal(200))
				var faceresp Face.FaceSimResp
				Expect(json.Unmarshal(resp.BodyToByte(), &faceresp)).Should(BeNil())
				Expect(faceresp.Code).To(Equal(0))
				Expect(len(faceresp.Result.Faces)).To(Equal(2))
				for _, face := range faceresp.Result.Faces {
					assert.CheckPts(face.Pts)
					Expect(face.Score).To(BeNumerically(">", 0.999))
				}
				Expect(faceresp.Result.Similarity).To(BeNumerically("<", 0.4))
				Expect(faceresp.Result.Similarity).To(BeNumerically(">=", 0.0))
				Expect(faceresp.Result.Same).To(Equal(false))
			})
		})
		Context("图片url错误", func() {
			var img = "faceage.jpg"
			var uri = E.Env.GetURIPrivate(img)
			err := biz.StoreUri(img, uri)
			if err != nil {
				panic(err)
			}
			It("图片url错误", func() {
				inputData := proto.NewArgusBatchReq(uri, nil, nil)
				inputData.Add("http", nil)
				By("图片请求Api:")
				resp, err := client.PostWithJson(path, inputData)
				Expect(err).Should(BeNil())
				By("检验api返回http code")
				Expect(resp.Status()).To(Equal(400))
				var resObj proto.ErrMessage
				Expect(json.Unmarshal(resp.BodyToByte(), &resObj)).Should(BeNil())
				Expect(resObj.Message).Should(Equal("uri not supported: http"))
			})
		})
	})
	Describe("反向用例", func() {
		var imgList = []string{
			"difftype-rgbbmp1.bmp",
			"difftype-tga1.tga",
			"big-201807121447.jpg",
			"big-201810181124.jpeg",
			"background-201812241100.jpg",
			"background-201812241059.jpg"}
		var imgSet = "test/image/normal/"
		var uriList []string
		for _, img := range imgList {
			uri := E.Env.GetURIPrivate(imgSet + img)
			err := biz.StoreUri(imgSet+img, uri)
			if err != nil {
				panic(err)
			}
			uriList = append(uriList, uri)
		}
		Context("Code 4000201", func() {
			It("normal", func() {
				inputData := proto.NewArgusBatchReq("rtmp", nil, nil)
				inputData.Add("rtmp", nil)
				By("图片请求Api:")
				resp, err := client.PostWithJson(path, inputData)
				Expect(err).Should(BeNil())
				By("检验api返回http code")
				var res proto.CodeErrorResp
				Expect(resp.Status()).To(Equal(400))
				Expect(json.Unmarshal(resp.BodyToByte(), &res)).Should(BeNil())
				Expect(res.Error).Should(Equal("uri not supported: rtmp"))
				Expect(res.Code).Should(Equal(4000201))
			})
		})
		Context("Code 4000203", func() {
			It("normal", func() {
				inputData := proto.NewArgusBatchReq("http://", nil, nil)
				inputData.Add("http://", nil)
				By("图片请求Api:")
				resp, err := client.PostWithJson(path, inputData)
				Expect(err).Should(BeNil())
				By("检验api返回http code")
				var res proto.CodeErrorResp
				Expect(resp.Status()).To(Equal(400))
				Expect(json.Unmarshal(resp.BodyToByte(), &res)).Should(BeNil())
				Expect(res.Error).Should(Equal("fetch uri failed: http://"))
				Expect(res.Code).Should(Equal(4000203))
			})
		})

		Context("Code 4150301", func() {
			It("normal", func() {
				inputData := proto.NewArgusBatchReq(uriList[0], nil, nil)
				inputData.Add(uriList[1], nil)
				By("图片请求Api:")
				resp, err := client.PostWithJson(path, inputData)
				Expect(err).Should(BeNil())
				By("检验api返回http code")
				var res proto.CodeErrorResp
				Expect(resp.Status()).To(Equal(415))
				Expect(json.Unmarshal(resp.BodyToByte(), &res)).Should(BeNil())
				Expect(res.Error).Should(Equal("image: unknown format"))
				Expect(res.Code).Should(Equal(4150301))
			})
		})
		Context("Code 4000302 :test1", func() {
			It("normal", func() {
				inputData := proto.NewArgusBatchReq(uriList[2], nil, nil)
				inputData.Add(uriList[2], nil)
				By("图片请求Api:")
				resp, err := client.PostWithJson(path, inputData)
				Expect(err).Should(BeNil())
				By("检验api返回http code")
				var res proto.CodeErrorResp
				Expect(resp.Status()).To(Equal(400))
				Expect(json.Unmarshal(resp.BodyToByte(), &res)).Should(BeNil())
				Expect(res.Error).Should(Equal("image is too large, should be in 4999x4999"))
				Expect(res.Code).Should(Equal(4000302))
			})
		})
		Context("Code 4000302 :test2", func() {
			It("normal", func() {
				inputData := proto.NewArgusBatchReq(uriList[3], nil, nil)
				inputData.Add(uriList[3], nil)
				By("图片请求Api:")
				resp, err := client.PostWithJson(path, inputData)
				Expect(err).Should(BeNil())
				By("检验api返回http code")
				var res proto.CodeErrorResp
				Expect(resp.Status()).To(Equal(400))
				Expect(json.Unmarshal(resp.BodyToByte(), &res)).Should(BeNil())
				Expect(res.Error).Should(Equal("image is too large, should be less than 10MB"))
				Expect(res.Code).Should(Equal(4000302))
			})
		})
		Context("Code 4000601", func() {
			It("normal", func() {
				inputData := proto.NewArgusBatchReq(uriList[4], nil, nil)
				inputData.Add(uriList[5], nil)
				By("图片请求Api:")
				resp, err := client.PostWithJson(path, inputData)
				Expect(err).Should(BeNil())
				By("检验api返回http code")
				var res proto.CodeErrorResp
				Expect(resp.Status()).To(Equal(400))
				Expect(json.Unmarshal(resp.BodyToByte(), &res)).Should(BeNil())
				Expect(res.Error).Should(Equal("found no face"))
				Expect(res.Code).Should(Equal(4000601))
			})
		})
	})
})

var _ = Describe("[sim]|[face]|[facexdetect]|/v1/eval/...face.image_sync.facex-detect", func() {
	evaltest.EvalFacexDetectTest("facex-detect", "facedetect", "image")
})

var _ = Describe("[sim]|[face]|[facexdetect]|/v1/eval/...face.image_sync.facex-feature", func() {
	evaltest.EvalFacexFeatureTest("facex-feature", "facefeature", "image")
})
