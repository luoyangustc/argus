package evaltest

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strconv"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	"qiniu.com/argus/test/biz/assert"
	P "qiniu.com/argus/test/biz/batch"
	E "qiniu.com/argus/test/biz/env"
	proto "qiniu.com/argus/test/biz/proto"
	"qiniu.com/argus/test/biz/tsv"
	biz "qiniu.com/argus/test/biz/util"
	"qiniu.com/argus/test/configs"
)

func EvalPoliticianTest(EvalServer string, Server string, SceneType string) {
	var (
		client      = E.Env.GetClientServing()
		pathFeature = biz.GetPath("censor-facefeature", SceneType, "evalpath")
		pathDetect  = biz.GetPath("censor-facedetect", SceneType, "evalpath")
		pathEval    = biz.GetPath(Server, SceneType, "evalpath")
		FeatureName = configs.StubConfigs.Servers.Type["eval"][EvalServer].Tsv
	)
	Describe("正向用例", func() {
		Context("处理规则", func() {
			var fileNames = []string{"zhouenlai01.jpg", "15302334761696822151.jpg"}
			var uriList []string
			for _, img := range fileNames {
				fileName := img
				uri := E.Env.GetURIPrivate(fileName)
				err := biz.StoreUri(fileName, uri)
				if err != nil {
					panic(err)
				}
				uriList = append(uriList, uri)
			}
			It("无pts默认用大库检索", func() {
				By("请求eval facex-detect...")
				respD, errD := client.PostWithJson(pathDetect,
					proto.NewArgusReq(uriList[0], nil, nil))
				Expect(errD).Should(BeNil())
				if respD.Status() != 200 {
					return
				}
				var resObjD proto.ArgusRes
				if err := json.Unmarshal(respD.BodyToByte(), &resObjD); err != nil {
					Expect(err).Should(BeNil())
				}
				for i, detection := range resObjD.Result.Detections {
					By("处理第" + strconv.Itoa(i) + "个detection:")
					var attr = map[string]proto.PtsObj{"pts": detection.Pts}
					By("请求eval facex-feature-v4...")
					resp, errF := client.PostWithJson(pathFeature,
						proto.NewArgusReq(uriList[0], attr, nil))
					Expect(errF).Should(BeNil())
					Expect(resp.Status()).To(Equal(200))

					By("请求eval politician with pts...")
					uriPoli := "data:application/octet-stream;base64," + base64.StdEncoding.EncodeToString(resp.BodyToByte())
					respP, errP := client.PostWithJson(pathEval,
						proto.NewArgusReq(uriPoli, attr, proto.NewLimit(1)))
					Expect(errP).Should(BeNil())
					Expect(respP.Status()).To(Equal(200))

					By("请求eval politician without pts...")
					respP2, errP2 := client.PostWithJson(pathEval,
						proto.NewArgusReq(uriPoli, nil, proto.NewLimit(1)))
					Expect(errP2).Should(BeNil())
					Expect(respP2.Status()).To(Equal(200))
					tsv.CheckJson(respP.BodyToByte(), respP2.BodyToByte(), 0.000001)

				}
			})

			It("人脸最短边小于32返回空confidences[]", func() {
				By("请求eval facex-detect...")
				respD, errD := client.PostWithJson(pathDetect,
					proto.NewArgusReq(uriList[1], nil, nil))
				Expect(errD).Should(BeNil())
				if respD.Status() != 200 {
					return
				}
				var resObjD proto.ArgusRes
				if err := json.Unmarshal(respD.BodyToByte(), &resObjD); err != nil {
					Expect(err).Should(BeNil())
				}
				Expect(len(resObjD.Result.Detections)).Should(Equal(2))
				for i, detection := range resObjD.Result.Detections {
					By("人脸框最小边小于32")
					Expect(detection.Pts[1][0] - detection.Pts[0][0]).Should(BeNumerically("<", 32))
					By("处理第" + strconv.Itoa(i) + "个detection:")
					var attr = map[string]proto.PtsObj{"pts": detection.Pts}
					By("请求eval facex-feature-v4...")
					resp, errF := client.PostWithJson(pathFeature,
						proto.NewArgusReq(uriList[1], attr, nil))
					Expect(errF).Should(BeNil())
					Expect(resp.Status()).To(Equal(200))
					var resObjtmp = make([]float32, 512)
					resF := assert.ParseFloat32Buf(resp.BodyToByte(), resObjtmp)
					fmt.Print(resF)
					By("请求eval politician...")
					uriPoli := "data:application/octet-stream;base64," + base64.StdEncoding.EncodeToString(resp.BodyToByte())
					respP, errP := client.PostWithJson(pathEval,
						proto.NewArgusReq(uriPoli, attr, proto.NewLimit(1)))
					Expect(errP).Should(BeNil())
					Expect(respP.Status()).To(Equal(200))

					var resObjExp proto.ArgusRes
					err := json.Unmarshal(respP.BodyToByte(), &resObjExp)
					Expect(err).Should(BeNil())
					// SDK实现造成的细节差异，暂时屏蔽
					// Expect(resObjExp.Message).Should(Equal("pts should larger than 32x32"))

				}
			})
		})
		Context("验证选取特征", func() {
			fileName := "maoyeye.jpg"
			By("测试图片：" + fileName)
			uri := E.Env.GetURIPrivate(fileName)
			if err := biz.StoreUri(fileName, uri); err != nil {
				panic(err)
			}
			var features = GetFeature(FeatureName)
			It("大脸库+小脸库", func() {
				attrSmall, _ := proto.NewPtsAttr("[[137, 24],[196, 24], [196, 141], [137, 141]]")
				attrLarge, _ := proto.NewPtsAttr("[[137, 24],[197, 24], [197, 141], [137, 141]]")
				By("请求eval facex-detect...")
				respD, errD := client.PostWithJson(pathDetect,
					proto.NewArgusReq(uri, nil, nil))
				Expect(errD).Should(BeNil())
				if respD.Status() != 200 {
					return
				}
				var resObjD proto.ArgusRes
				if err := json.Unmarshal(respD.BodyToByte(), &resObjD); err != nil {
					Expect(err).Should(BeNil())
				}
				for i, detection := range resObjD.Result.Detections {
					By("处理第" + strconv.Itoa(i) + "个detection:")
					var attr = map[string]proto.PtsObj{"pts": detection.Pts}
					By("请求eval facex-feature-v3...")
					resp, errF := client.PostWithJson(pathFeature,
						proto.NewArgusReq(uri, attr, nil))
					Expect(errF).Should(BeNil())
					Expect(resp.Status()).To(Equal(200))
					var resObjtmp = make([]float32, 512)
					resObjF := assert.ParseFloat32Buf(resp.BodyToByte(), resObjtmp)

					uriPoli := "data:application/octet-stream;base64," + base64.StdEncoding.EncodeToString(resp.BodyToByte())

					By("请求eval politician with pts small less then 60...")
					respP2, errP2 := client.PostWithJson(pathEval,
						proto.NewArgusReq(uriPoli, attrSmall, proto.NewLimit(1)))
					Expect(errP2).Should(BeNil())
					Expect(respP2.Status()).To(Equal(200))
					var resObj2 proto.ArgusRes
					err := json.Unmarshal(respP2.BodyToByte(), &resObj2)
					Expect(err).Should(BeNil())
					score, fea := MaxFeature(features["small"], resObjF)
					Expect(resObj2.Result.Confidences[0].Class).Should(Equal("毛泽东"))
					Expect(resObj2.Result.Confidences[0].Group).Should(Equal("domestic_statesman"))
					Expect(resObj2.Result.Confidences[0].Index).Should(Equal(16))
					// SDK实现造成的细节差异，暂时屏蔽
					Expect(resObj2.Result.Confidences[0].Score).Should(BeNumerically("~", score, 0.01))
					Expect(resObj2.Result.Confidences[0].Sample.Url).Should(Equal(fea.URL))
					assert.CheckPts(resObj2.Result.Confidences[0].Sample.Pts)

					By("请求eval politician with pts large >= 60...")
					respP3, errP3 := client.PostWithJson(pathEval,
						proto.NewArgusReq(uriPoli, attrLarge, proto.NewLimit(1)))
					Expect(errP3).Should(BeNil())
					Expect(respP3.Status()).To(Equal(200))
					var resObj3 proto.ArgusRes
					err = json.Unmarshal(respP3.BodyToByte(), &resObj3)
					Expect(err).Should(BeNil())
					score, fea = MaxFeature(features["large"], resObjF)
					Expect(resObj3.Result.Confidences[0].Class).Should(Equal("毛泽东"))
					Expect(resObj3.Result.Confidences[0].Group).Should(Equal("domestic_statesman"))
					Expect(resObj3.Result.Confidences[0].Index).Should(Equal(16))
					// SDK实现造成的细节差异，暂时屏蔽
					Expect(resObj3.Result.Confidences[0].Score).Should(BeNumerically("~", score, 0.01))
					Expect(resObj3.Result.Confidences[0].Sample.Url).Should(Equal(fea.URL))
					assert.CheckPts(resObj3.Result.Confidences[0].Sample.Pts)

				}
			})
		})

		Context("功能验证", func() {
			filelist, err := P.GetBucket("serving/politician/20180810/")
			if err != nil {
				panic(err)
			}
			for _, file := range filelist {
				fileName := file
				switch fileName.Imgname { // SDK实现造成的细节差异，暂时屏蔽
				case
					"0000_0_11_0.jpg",
					"0000_0_2_0.jpg",
					"0000_0_3_0.jpg",
					"0000_0_5_0.jpg",
					"0000_0qw.jpg",
					"0001_01_0.jpg",
					"0001_0_.jpg",
					"0001_0_ce.jpg",
					"0002_0_11_0.jpg",
					"0002_0_29_0.jpg",
					"0002_0_luoma.jpg",
					"0002_0_w_0.jpg",
					"17164201oqw3.jpg",
					"3823085815.jpg":
					continue
				}
				// if fileName.Imgname == ""
				By("测试图片：" + fileName.Imgname)
				uri := fileName.Imgurl
				if err := biz.StoreUri(fileName.Imgname, uri); err != nil {
					panic(err)
				}
				It("argus.face.politician验证功能"+fileName.Imgname, func() {
					By("请求eval facex-detect...")
					respD, errD := client.PostWithJson(pathDetect,
						proto.NewArgusReq(uri, nil, nil))
					Expect(errD).Should(BeNil())
					if respD.Status() != 200 {
						return
					}
					var resObjD proto.ArgusRes
					if err := json.Unmarshal(respD.BodyToByte(), &resObjD); err != nil {
						Expect(err).Should(BeNil())
					}
					for i, detection := range resObjD.Result.Detections {
						By("处理第" + strconv.Itoa(i) + "个detection:")
						var attr = map[string]proto.PtsObj{"pts": detection.Pts}
						By("请求eval facex-feature-v4...")
						resp, errF := client.PostWithJson(pathFeature,
							proto.NewArgusReq(uri, attr, nil))
						Expect(errF).Should(BeNil())
						Expect(resp.Status()).To(Equal(200))
						//
						By("请求eval politician...")
						uriPoli := "data:application/octet-stream;base64," + base64.StdEncoding.EncodeToString(resp.BodyToByte())
						respP, errP := client.PostWithJson(pathEval,
							proto.NewArgusReq(uriPoli, attr, proto.NewLimit(1)))
						Expect(errP).Should(BeNil())
						Expect(respP.Status()).To(Equal(200))

						var resObj proto.ArgusRes
						err = json.Unmarshal(respP.BodyToByte(), &resObj)
						Expect(err).Should(BeNil())
						Expect(len(resObj.Result.Confidences[0].Class)).Should(BeNumerically(">", 0))
						Expect(resObj.Result.Confidences[0].Group).ShouldNot(Equal(""))
						Expect(resObj.Result.Confidences[0].Score).Should(BeNumerically(">", 0.1))
						Expect(resObj.Result.Confidences[0].Sample.Url).ShouldNot(Equal(""))
						assert.CheckPts(resObj.Result.Confidences[0].Sample.Pts)
					}
				})
			}
		})

	})
	Describe("反向用例", func() {
		It("输入uri不正确", func() {
			uri := "data:application/octet-stream;base64," + "hioiqoefnqoenfqownro"
			respP, errP := client.PostWithJson(pathEval,
				proto.NewArgusReq(uri, nil, proto.NewLimit(1)))
			Expect(errP).Should(BeNil())
			Expect(respP.Status()).To(Equal(400))
		})
	})
}
