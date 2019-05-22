package test

import (
	"encoding/json"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	EvalTest "qiniu.com/argus/service/service/image/terror/image_sync/test/evaltest"
	"qiniu.com/argus/test/biz/assert"
	P "qiniu.com/argus/test/biz/batch"
	E "qiniu.com/argus/test/biz/env"
	"qiniu.com/argus/test/biz/proto"
	biz "qiniu.com/argus/test/biz/util"
)

var _ = Describe("[argus]|[terror]|/v1/terror", func() {
	var (
		server                = "terror"
		pathMixup     string  = biz.GetPath("evalTerrorMixup", "image", "evalpath")
		pathDetect    string  = biz.GetPath("evalTerrorDetect", "image", "evalpath")
		path                  = biz.GetPath(server, "image", "path")
		client                = E.Env.GetClientArgus()
		clientServing         = E.Env.GetClientServing()
		params                = map[string]bool{"detail": true}
		precision     float64 = 0.0001
	)
	Describe("暴恐串联ATLAB-9013", func() {
		Context("terror-mixup过滤", func() {
			image := []string{"0_bk_bloodiness_human_0227_650.jpg",
				"1_bk_bomb_fire_0227_485.jpg",
				"2_bk_bomb_smoke_0227_301.jpg",
				"3_bk_bomb_vehicle_0227_374.jpg",
				"46_background_with_people_0227_441.jpg",
				"7_bk_march_banner_0227_309.jpg",
				"n03763968_10547.jpg"}
			imageSet := "serving/terror-mixup/set20181108/"
			for _, file := range image {
				fileName := imageSet + file
				uri := E.Env.GetURIPrivate(fileName)
				if err := biz.StoreUri(fileName, uri); err != nil {
					panic(err)
				}
				It("测试图片:"+fileName, func() {
					resp, err := client.PostWithJson(path,
						proto.NewArgusReq(uri, nil, params))
					Expect(err).Should(BeNil())
					Expect(resp.Status()).To(Equal(200))
					respServ, errServ := clientServing.PostWithJson(pathMixup,
						proto.NewArgusReq(uri, nil, nil))
					Expect(errServ).Should(BeNil())
					Expect(respServ.Status()).To(Equal(200))
					CheckConfidences("endpoint", resp.BodyToByte(), respServ.BodyToByte(), precision)
				})
			}
		})
		Context("terror-detect过滤", func() {
			image := []string{
				"20181129-terror-islamicflag.jpg",
				"20170930_islamic_flag_1382.jpg",
				"20181129-terror-knives.jpeg"}
			imageSet := "serving/terror-mixup/set20181108/"
			for _, file := range image {
				fileName := imageSet + file
				uri := E.Env.GetURIPrivate(fileName)
				if err := biz.StoreUri(fileName, uri); err != nil {
					panic(err)
				}
				It("测试图片:"+fileName, func() {
					resp, err := client.PostWithJson(path,
						proto.NewArgusReq(uri, nil, params))
					Expect(err).Should(BeNil())
					Expect(resp.Status()).To(Equal(200))
					respServ, errServ := clientServing.PostWithJson(pathDetect,
						proto.NewArgusReq(uri, nil, nil))
					Expect(errServ).Should(BeNil())
					Expect(respServ.Status()).To(Equal(200))
					CheckDetections(resp.BodyToByte(), respServ.BodyToByte(), precision)
				})
			}
		})
		Context("terror-mixup未过滤，terror-detect结果为[]", func() {
			image := []string{
				"0_bk_bloodiness_human_0227_669.jpg",
				"47_background_no_people_0227_131.jpg",
				"46_background_with_people_0227_756.jpg",
				"7_bk_march_banner_0227_336.jpg"}
			imageSet := "serving/terror-mixup/set20181108/"
			for _, file := range image {
				fileName := imageSet + file
				uri := E.Env.GetURIPrivate(fileName)
				if err := biz.StoreUri(fileName, uri); err != nil {
					panic(err)
				}
				It("测试图片:"+fileName, func() {
					resp, err := client.PostWithJson(path,
						proto.NewArgusReq(uri, nil, params))
					Expect(err).Should(BeNil())
					Expect(resp.Status()).To(Equal(200))
					respServ, errServ := clientServing.PostWithJson(pathMixup,
						proto.NewArgusReq(uri, nil, nil))
					Expect(errServ).Should(BeNil())
					Expect(respServ.Status()).To(Equal(200))
					CheckConfidences("terror-detect", resp.BodyToByte(), respServ.BodyToByte(), precision)
					respServD, errServD := clientServing.PostWithJson(pathDetect,
						proto.NewArgusReq(uri, nil, nil))
					Expect(errServD).Should(BeNil())
					Expect(respServD.Status()).To(Equal(200))
					var resObjD proto.ArgusRes
					if errD := json.Unmarshal(respServD.BodyToByte(), &resObjD); err != nil {
						Expect(errD).Should(BeNil())
					}
					Expect(len(resObjD.Result.Detections)).To(Equal(0))
				})
			}
		})
	})
	Describe("功能测试", func() {
		Context("不同类别验证", func() {
			var params = map[string]bool{"detail": true}
			imgSet := "argus/terror/test/"
			filelist := []string{"bomb_self_burning.jpeg",
				"bloodiness.jpeg",
				"anime_knives.jpg",
				"anime_bloodiness.jpeg",
				"anime_guns.jpeg",
				"guns.jpg",
				"illegal_flag.jpg",
				"knives.jpg",
				"bk_more_guns.jpg",
				"fight_person.jpg",
				"march_crowed.jpg",
				"fight_police.jpg",
				"special_characters.jpg",
				"terror_complex.png",
				"normaltest.jpeg"}
			labellist := []int{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0}
			classlist := []string{"self_burning",
				"bloodiness",
				"anime_knives",
				"anime_bloodiness",
				"anime_guns",
				"guns",
				"illegal_flag",
				"knives",
				"guns",
				"fight_person",
				"march_crowed",
				"fight_police",
				"special_characters",
				"normal",
				"normal"}
			for i, file := range filelist {
				fileName := imgSet + file
				uri := E.Env.GetURIPrivate(fileName)
				if err := biz.StoreUri(fileName, uri); err != nil {
					panic(err)
				}
				ii := i
				It("测试图片:"+fileName, func() {
					//调用api获取最新结果
					// file 总是pulpfiles的最后一个数据，故用fileName
					// By("测试图片：" + fileName)
					resp, err := client.PostWithJson(path,
						proto.NewArgusReq(uri, nil, params))
					Expect(err).Should(BeNil())
					Expect(resp.Status()).To(Equal(200))
					//以下为验证结果
					var resObj proto.ArgusRes
					// Expect(resObj.Result.Class).To(Equal(labellist))
					Expect(json.Unmarshal(resp.BodyToByte(), &resObj)).Should(BeNil())
					Expect(resObj.Result.Label).Should(Equal(labellist[ii]))
					Expect(resObj.Result.Class).Should(Equal(classlist[ii]))
				})
			}
			Context("纯黑图片测试", func() {
				fileName := "black.jpg"
				imgSet = "argus/terror/test/"
				url := E.Env.GetURIPrivate(imgSet + fileName)
				if err := biz.StoreUri(imgSet+fileName, url); err != nil {
					panic(err)
				}
				It("图片测试"+fileName, func() {
					resp, err := client.PostWithJson(path,
						proto.NewArgusReq(url, nil, params))
					Expect(err).Should(BeNil())
					Expect(resp.Status()).To(Equal(200))
					var resObj proto.ArgusRes
					err = json.Unmarshal(resp.BodyToByte(), &resObj)
					Expect(err).Should(BeNil())
					Expect(resObj.Result.Score).To(BeNumerically(">", 0.4))
					Expect(resObj.Result.Review).To(Equal(true))
				})
			})
			Context("测试review", func() {
				var fileNames = []string{"weiboimg-2017-11-17-09-56-aHR0cHM6Ly93dzEuc2luYWltZy5jbi9vcmozNjAvNTNhZTBiNzBqdzFlaml5YjM2cHJmajIwbnAwZGN3aDEuanBn.jpg",
					"n03041632_45090.jpg"}
				imgSet := "serving/terror-detect/set20180416/"
				var uriList []string
				for _, img := range fileNames {
					uri := E.Env.GetURIPrivate(imgSet + img)
					err := biz.StoreUri(imgSet+img, uri)
					if err != nil {
						panic(err)
					}
					uriList = append(uriList, uri)
				}
				It("测试need review", func() {
					resp, err := client.PostWithJson(path,
						proto.NewArgusReq(uriList[0], nil, params))
					Expect(err).Should(BeNil())
					Expect(resp.Status()).To(Equal(200))
					var resObj proto.ArgusRes
					err = json.Unmarshal(resp.BodyToByte(), &resObj)
					Expect(err).Should(BeNil())
					Expect(resObj.Result.Score).To(BeNumerically("<", 0.85))
					Expect(resObj.Result.Review).To(Equal(true))
				})
				It("测试not review", func() {
					resp, err := client.PostWithJson(path,
						proto.NewArgusReq(uriList[1], nil, params))
					Expect(err).Should(BeNil())
					Expect(resp.Status()).To(Equal(200))
					var resObj proto.ArgusRes
					err = json.Unmarshal(resp.BodyToByte(), &resObj)
					Expect(err).Should(BeNil())
					Expect(resObj.Result.Score).To(BeNumerically(">", 0.85))
					Expect(resObj.Result.Review).To(Equal(false))
				})
			})
		})

		Context("detail功能验证", func() {
			var params = map[string]bool{"detail": true}
			//功能性验证
			filelist := []string{"bloodiness.jpeg", "march_crowed.jpeg", "bomb_self_burning.jpeg", "normaltest.jpeg"}
			labellist := []int{1, 0, 1, 0}
			for i, file := range filelist {
				// file 总是pulpfiles的最后一个数据，故用fileName
				fileName := file
				ii := i
				By("测试图片：" + fileName)
				uri := E.Env.GetURIPrivate(fileName)
				if err := biz.StoreUri(fileName, uri); err != nil {
					panic(err)
				}
				It("argus.terror验证功能", func() {
					//调用api获取最新结果
					resp, err := client.PostWithJson(path,
						proto.NewArgusReq(uri, nil, params))
					Expect(err).Should(BeNil())
					Expect(resp.Status()).To(Equal(200))
					//以下为验证结果
					var resObj proto.ArgusRes

					Expect(json.Unmarshal(resp.BodyToByte(), &resObj)).Should(BeNil())
					Expect(resObj.Result.Label).Should(Equal(labellist[ii]))
					Expect(resObj.Result.Class).NotTo(Equal(""))
					// 无detail
					resp, err = client.PostWithJson(path,
						proto.NewArgusReq(uri, nil, nil))
					Expect(err).Should(BeNil())
					Expect(resp.Status()).To(Equal(200))
					//以下为验证结果
					var resObj2 proto.ArgusRes

					Expect(json.Unmarshal(resp.BodyToByte(), &resObj2)).Should(BeNil())
					Expect(resObj2.Result.Label).Should(Equal(labellist[ii]))
					Expect(resObj2.Result.Class).To(Equal(""))
				})
			}
			It("String Detail", func() {
				var StrParams = map[string]string{"detail": "true"}
				resp, err := client.PostWithJson(path,
					proto.NewArgusReq(E.Env.GetURIPrivate(filelist[3]), nil, StrParams))
				Expect(err).Should(BeNil())
				Expect(resp.Status()).To(Equal(599))
				msg := resp.ResponseBodyAsJson()["error"].(string)
				Expect(msg).Should(Equal("json: cannot unmarshal string into Go struct field .detail of type bool"))
			})
		})
		Context("特殊图片", func() {
			//功能性验证
			filelist := []string{"upload.gif", "big.jpg", "bloodiness.webp", "bloodiness.bmp"}
			for _, file := range filelist {
				fileName := file
				By("测试图片：" + fileName)
				uri := E.Env.GetURIPrivate(fileName)
				if err := biz.StoreUri(fileName, uri); err != nil {
					panic(err)
				}
				It("argus.terror 验证功能", func() {
					//调用api获取最新结果
					// file 总是pulpfiles的最后一个数据，故用fileName
					resp, err := client.PostWithJson(path,
						proto.NewArgusReq(uri, nil, nil))
					Expect(err).Should(BeNil())
					Expect(resp.Status()).To(Equal(200))

				})
			}
		})
	})
	Describe("测试图片库", func() {
		Context("测试图片库", func() {
			imgList, err := P.BatchGetImg("normal")
			if err != nil {
				panic(err)
			}
			imgCh := make(chan string, len(imgList))
			for _, img := range imgList {
				if img.Imgname == "difftype-4ch.webp" {
					continue
				}
				err = biz.StoreUri(img.Imgname, img.Imgurl)
				if err != nil {
					panic(err)
				}
				imgCh <- img.Imgurl
				It("测试图片:"+img.Imgname, func() {
					// By("测试图片：" + img.Imgname)
					url, ok := <-imgCh
					Expect(ok).To(Equal(true))
					resp, err := client.PostWithJson(path,
						proto.NewArgusReq(url, nil, nil))
					Expect(err).Should(BeNil())
					Expect(resp.Status()).To(Or(Equal(200), BeNumerically("<", 500)))
					var resObj proto.ArgusRes
					Expect(json.Unmarshal(resp.BodyToByte(), &resObj)).Should(BeNil())
					if resp.Status() == 200 {
						if resObj.Result.Score < 0.85 {
							Expect(resObj.Result.Review).To(Equal(true))
						} else {
							Expect(resObj.Result.Review).To(Equal(false))
						}
					}

				})
			}
		})
		Context("Check Code", func() {
			assert.CheckImageCode(client, path, nil)
		})
	})
})

var _ = Describe("[eval]|[TerrorDetect]|/v1/eval/...terror.image_sync.evalTerrorDetect", func() {
	EvalTest.EvalTerrorDetectTest("TerrorDetect", "evalTerrorDetect", "image")
})

var _ = Describe("[eval]|[TerrorMixup]|/v1/eval/...terror.image_sync.evalTerrorMixup", func() {
	EvalTest.EvalTerrorMixupTest("TerrorMixup", "evalTerrorMixup", "image")
})
