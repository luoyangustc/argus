package evaltest

import (
	"encoding/json"
	"strconv"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	"qiniu.com/argus/test/biz/assert"
	P "qiniu.com/argus/test/biz/batch"
	E "qiniu.com/argus/test/biz/env"
	"qiniu.com/argus/test/biz/proto"
	T "qiniu.com/argus/test/biz/tsv"
	biz "qiniu.com/argus/test/biz/util"
	"qiniu.com/argus/test/configs"
)

func CheckDetection(respone []byte, exp []byte, precision float64) {
	//以下为验证结果
	var resObj proto.ArgusRes
	if err := json.Unmarshal(respone, &resObj); err != nil {
		Expect(err).Should(BeNil())
	}
	var expObj []proto.ArgusDetection
	if err := json.Unmarshal(exp, &expObj); err != nil {
		Expect(err).Should(BeNil())
	}
	By("检验detection个数")
	Expect(len(resObj.Result.Detections)).To(Equal(len(expObj)))
	for k, ResultRun := range resObj.Result.Detections {
		By("检验位置：" + strconv.Itoa(k))

		By("检验index")
		Expect(ResultRun.Index).To(Equal(expObj[k].Index))
		By("检验score")
		Expect(ResultRun.Score).To(BeNumerically("~", expObj[k].Score, precision))
		By("检验Class")
		Expect(ResultRun.Class).To(Equal(expObj[k].Class))
		By("检验AreaRatio")
		Expect(ResultRun.AreaRatio).To(BeNumerically("~", expObj[k].AreaRatio, precision))
		By("检验Pts")
		for i, pt := range expObj[k].Pts {
			for j, p := range pt {
				Expect(ResultRun.Pts[i][j]).To(Equal(p))
			}
		}
		By("检验Orientation")
		Expect(ResultRun.Orientation).To(Equal(expObj[k].Orientation))
		By("检验Qscore")

		Expect(ResultRun.Qscore.Blur).To(BeNumerically("~", expObj[k].Qscore.Blur, precision))
		Expect(ResultRun.Qscore.Clear).To(BeNumerically("~", expObj[k].Qscore.Clear, precision))
		Expect(ResultRun.Qscore.Neg).To(BeNumerically("~", expObj[k].Qscore.Neg, precision))
		Expect(ResultRun.Qscore.Pose).To(BeNumerically("~", expObj[k].Qscore.Pose, precision))
		var quality = "small"
		maxScore := 0.0
		if ResultRun.Qscore.Blur > maxScore {
			maxScore = ResultRun.Qscore.Blur
			quality = "blur"
		}
		if ResultRun.Qscore.Clear > maxScore {
			maxScore = ResultRun.Qscore.Clear
			quality = "clear"
		}
		if ResultRun.Qscore.Cover > maxScore {
			maxScore = ResultRun.Qscore.Cover
			quality = "cover"
		}
		if ResultRun.Qscore.Neg > maxScore {
			maxScore = ResultRun.Qscore.Neg
			quality = "neg"
		}
		if ResultRun.Qscore.Pose > maxScore {
			maxScore = ResultRun.Qscore.Pose
			quality = "pose"
		}
		if quality == "clear" && maxScore < 0.6 {
			quality = "blur"
		}
		if ResultRun.Quality != "" {
			Expect(ResultRun.Quality).To(Equal(quality))
			// Expect(ResultRun.Quality).To(Equal(expObj[k].Quality))
		}
	}
}

func EvalFacexDetectTest(EvalServer string, Server string, SceneType string) {
	var clientServing = E.Env.GetClientServing()
	var path = biz.GetPath(Server, SceneType, "evalpath")
	var limit = proto.NewLimit(1)
	var tsv = T.NewTsv(configs.StubConfigs.Servers.Type["eval"][EvalServer].Tsv,
		configs.StubConfigs.Servers.Type["eval"][EvalServer].Set, path, configs.StubConfigs.Servers.Type["eval"][EvalServer].Precision,
		proto.NewLimit(configs.StubConfigs.Servers.Type["eval"][EvalServer].Limit))

	Describe("eval.facex-detect tsv", func() {
		Context("测试人脸检测tsv", func() {
			T.TsvTest(clientServing, tsv, CheckDetection)
		})
		Context("功能验证", func() {
			//功能性验证
			imgSet := "test/image/face-detect/"
			pulpfiles := []string{"more-face.jpeg", "no-face.jpeg", "facex-detect-panic.webp"}
			for _, file := range pulpfiles {
				fileName := imgSet + file
				uri := E.Env.GetURIPrivate(fileName)
				if err := biz.StoreUri(fileName, uri); err != nil {
					panic(err)
				}
				It("arguspulp－验证功能", func() {
					//调用api获取最新结果
					// file 总是pulpfiles的最后一个数据，故用fileName
					resp, err := clientServing.PostWithJson(path,
						proto.NewArgusReq(uri, nil, limit))
					Expect(err).Should(BeNil())
					Expect(resp.Status()).To(Equal(200))
				})
			}
		})
		Context("test 4 faces", func() {
			img := "face10_LARGEST.jpg"
			url := E.Env.GetURIPrivate(img)
			if err := biz.StoreUri(img, url); err != nil {
				panic(err)
			}
			It("test 4 faces", func() {
				// 未上线

				resp, err := clientServing.PostWithJson(path,
					proto.NewArgusReq(url, nil, nil))
				Expect(err).Should(BeNil())
				Expect(resp.Status()).To(Equal(200))
				var resObjD proto.ArgusRes
				if err := json.Unmarshal(resp.BodyToByte(), &resObjD); err != nil {
					Expect(err).Should(BeNil())
				}
				Expect(len(resObjD.Result.Detections)).To(Equal(4))
			})
		})
		XContext("人脸朝向", func() {
			img := "face019.jpg"
			url := E.Env.GetURIPrivate(img)
			if err := biz.StoreUri(img, url); err != nil {
				panic(err)
			}
			It("人脸朝向验证", func() {
				resp, err := clientServing.PostWithJson(path,
					proto.NewArgusReq(url, nil, nil))
				Expect(err).Should(BeNil())
				Expect(resp.Status()).To(Equal(200))
				var resObjD proto.ArgusRes
				if err := json.Unmarshal(resp.BodyToByte(), &resObjD); err != nil {
					Expect(err).Should(BeNil())
				}

				Expect(len(resObjD.Result.Detections)).To(Equal(6))
				for _, Detection := range resObjD.Result.Detections {
					Expect(Detection.Score).Should(BeNumerically(">", 0.5))
					assert.CheckPts(Detection.Pts)
				}
				Expect(resObjD.Result.Detections[0].Orientation).To(Equal("up"))
				Expect(resObjD.Result.Detections[1].Orientation).To(Equal("up"))
				Expect(resObjD.Result.Detections[2].Orientation).To(Equal("down"))
				Expect(resObjD.Result.Detections[3].Orientation).To(Equal("right"))
				Expect(resObjD.Result.Detections[4].Orientation).To(Equal("left"))
				Expect(resObjD.Result.Detections[5].Orientation).To(Equal("up_right"))
			})
		})
		XContext("人脸质量度", func() {
			img := "face16_SMALLSIXE.jpg"
			url := E.Env.GetURIPrivate(img)
			if err := biz.StoreUri(img, url); err != nil {
				panic(err)
			}
			It("人脸质量类别验证", func() {
				resp, err := clientServing.PostWithJson(path,
					proto.NewArgusReq(url, nil, nil))
				Expect(err).Should(BeNil())
				Expect(resp.Status()).To(Equal(200))
				var resObjD proto.ArgusRes
				if err := json.Unmarshal(resp.BodyToByte(), &resObjD); err != nil {
					Expect(err).Should(BeNil())
				}
				Expect(len(resObjD.Result.Detections)).To(Equal(1))
				Expect(resObjD.Result.Detections[0].Quality).To(Equal("small"))
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
					resp, err := clientServing.PostWithJson(path,
						proto.NewArgusReq(imgUrl, nil, nil))
					Expect(err).Should(BeNil())
					Expect(resp.Status()).To(Or(Equal(200), BeNumerically("<", 500)))
					var resObj proto.ArgusRes
					err = json.Unmarshal(resp.BodyToByte(), &resObj)
					Expect(err).Should(BeNil())
				})
			}
		})
		Context("face", func() {
			imgList, err := P.BatchGetImg("face")
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
					resp, err := clientServing.PostWithJson(path,
						proto.NewArgusReq(imgUrl, nil, nil))
					Expect(err).Should(BeNil())
					Expect(resp.Status()).To(Or(Equal(200), BeNumerically("<", 500)))
					var resObj proto.ArgusRes
					err = json.Unmarshal(resp.BodyToByte(), &resObj)
					Expect(err).Should(BeNil())
				})
			}
		})
	})
}
