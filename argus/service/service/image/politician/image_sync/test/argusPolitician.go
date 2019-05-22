package test

import (
	"encoding/base64"
	"encoding/json"
	"strconv"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"qiniu.com/argus/service/service/image/politician"
	EvalTest "qiniu.com/argus/service/service/image/politician/image_sync/test/evaltest"
	"qiniu.com/argus/test/biz/assert"
	P "qiniu.com/argus/test/biz/batch"
	E "qiniu.com/argus/test/biz/env"
	proto "qiniu.com/argus/test/biz/proto"
	biz "qiniu.com/argus/test/biz/util"
)

var _ = Describe("[argus]|[politician]|/v1/face/search/politician", func() {
	var clientServing = E.Env.GetClientServing()
	var clientArgus = E.Env.GetClientArgus()
	var server = "politician"
	var path = biz.GetPath(server, "image", "path")
	var pathFeature = biz.GetPath("censor-facefeature", "image", "evalpath")
	var pathDetect = biz.GetPath("censor-facedetect", "image", "evalpath")
	var pathEval = biz.GetPath(server, "image", "evalpath")
	Describe("功能测试", func() {

		Context("功能验证", func() {
			//功能性验证
			// 图片zhouenlai-3.jpg，有小人脸，大人脸，需保留验证
			// SDK实现造成的细节差异，暂时屏蔽
			// filelist := []string{"zhouenlai-3.jpg", "zhouenlai01.jpg", "maoyeye.jpg", "15302334761696822151.jpg", "upload.gif", "sexy001.gif", "sexy002.gif", "luhan.jpeg"}
			filelist := []string{"zhouenlai01.jpg", "maoyeye.jpg", "15302334761696822151.jpg", "upload.gif", "sexy001.gif", "sexy002.gif"}
			// labellist := []int{1, 1, 1, 0}
			for _, file := range filelist {
				fileName := file
				// ii := i
				By("测试图片：" + fileName)
				uri := E.Env.GetURIPrivate(fileName)
				if err := biz.StoreUri(fileName, uri); err != nil {
					panic(err)
				}
				It("argus.face.politician验证功能"+fileName, func() {
					//调用api获取最新结果
					// file 总是pulpfiles的最后一个数据，故用fileName
					By("请求argus politician...")
					resp, err := clientArgus.PostWithJson(path,
						proto.NewArgusReq(uri, nil, nil))
					Expect(err).Should(BeNil())
					Expect(resp.Status()).To(Equal(200))
					var resObj politician.FaceSearchResp //proto.PoliticianArgusResponse
					err = json.Unmarshal(resp.BodyToByte(), &resObj)
					Expect(err).Should(BeNil())
					//
					By("请求eval facex-detect...")
					respD, errD := clientServing.PostWithJson(pathDetect,
						proto.NewArgusReq(uri, nil, nil))
					Expect(errD).Should(BeNil())
					if respD.Status() != 200 {
						Expect(resObj.Message).Should(Equal("No valid face info detected"))
					}
					var resObjD proto.ArgusRes
					if err := json.Unmarshal(respD.BodyToByte(), &resObjD); err != nil {
						Expect(err).Should(BeNil())
					}
					for i, detection := range resObjD.Result.Detections {
						By("处理第" + strconv.Itoa(i) + "个detection:")
						var attr = map[string]proto.PtsObj{"pts": detection.Pts}
						pts := resObjD.Result.Detections[i].Pts
						if pts[1][0]-pts[0][0] < 32 || pts[3][1]-pts[0][1] < 32 || detection.Score < 0.5 {
							continue
						}
						By("请求eval facex-feature-v4...")
						resp, errF := clientServing.PostWithJson(pathFeature,
							proto.NewArgusReq(uri, attr, nil))
						Expect(errF).Should(BeNil())
						Expect(resp.Status()).To(Equal(200))
						//
						By("请求eval politician...")
						uriPoli := "data:application/octet-stream;base64," + base64.StdEncoding.EncodeToString(resp.BodyToByte())
						respP, errP := clientServing.PostWithJson(pathEval,
							proto.NewArgusReq(uriPoli, attr, proto.NewLimit(1)))
						Expect(errP).Should(BeNil())
						Expect(respP.Status()).To(Equal(200))

						var resObjExp proto.ArgusRes
						err = json.Unmarshal(respP.BodyToByte(), &resObjExp)
						Expect(err).Should(BeNil())
						Expect(resObj.Result.Detections[i].BoundingBox.Pts).Should(Equal(resObjD.Result.Detections[i].Pts))
						Expect(resObj.Result.Detections[i].BoundingBox.Score).Should(BeNumerically("~", resObjD.Result.Detections[i].Score, 0.000001))
						// Expect(resObj.Result.Detections[i].Value.Score).Should(BeNumerically("~", resObjExp.Result.Confidences[0].Score, 0.000001))
						if resObj.Result.Detections[i].Value.Score > 0.375 {

							Expect(resObj.Result.Detections[i].Value.Name).Should(Equal(resObjExp.Result.Confidences[0].Class))
							Expect(resObj.Result.Detections[i].Value.Group).Should(Equal(resObjExp.Result.Confidences[0].Group))
							Expect(resObj.Result.Detections[i].Sample.Pts).Should(Equal(resObjExp.Result.Confidences[0].Sample.Pts))
							Expect(resObj.Result.Detections[i].Sample.URL).Should(Equal(resObjExp.Result.Confidences[0].Sample.Url))
						} else {
							Expect(resObj.Result.Detections[i].Value.Name).Should(Equal(""))
						}
						if resObj.Result.Detections[i].Value.Score > 0.4 || resObj.Result.Detections[i].Value.Score < 0.35 {
							Expect(resObj.Result.Detections[i].Value.Review).Should(Equal(false))
						} else if resObj.Result.Detections[i].Value.Score > 0.35 {
							Expect(resObj.Result.Detections[i].Value.Review).Should(Equal(true))
						}

					}
				})
			}
		})
		Context("特殊图片格式", func() {
			fileList := []string{"pulpsexy.webp", "test-pulp7.bmp"}
			for _, file := range fileList {
				fileName := file
				By("测试图片: " + fileName)
				uri := E.Env.GetURIPrivate(fileName)
				if err := biz.StoreUri(fileName, uri); err != nil {
					panic(err)
				}
				It("特殊图片 "+fileName, func() {
					resp, err := clientArgus.PostWithJson(path, proto.NewArgusReq(uri, nil, nil))
					Expect(err).Should(BeNil())
					Expect(resp.Status()).To(Equal(200))
				})
			}
		})
	})
	Describe("异常流测试", func() {
		Context("图片库测试", func() {
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
				})
			}
		})
		Context("Check Code", func() {
			assert.CheckImageCode(clientArgus, path, nil)
		})
	})
})

var _ = Describe("[eval]|[politician]|/v1/eval/...politician.image_sync.evalPolitician", func() {
	EvalTest.EvalPoliticianTest("Politician", "politician", "image")
})
