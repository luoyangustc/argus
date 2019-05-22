package test

import (
	"encoding/json"
	"fmt"

	"qiniu.com/argus/test/biz/assert"
	"qiniu.com/argus/test/biz/proto"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	Tcomplex "qiniu.com/argus/service/service/image/terror_complex"
	P "qiniu.com/argus/test/biz/batch"
	E "qiniu.com/argus/test/biz/env"
	biz "qiniu.com/argus/test/biz/util"
	"qiniu.com/argus/test/configs"
)

var _ = Describe("[argus]|[terror-complex]|/v1/terror/complex", func() {
	var server = "terror_complex"
	var path = biz.GetPath(server, "image", "path") //"/v1/terror/complex"
	var imgSet = configs.StubConfigs.Servers.Type["image"][server].Set
	var client = E.Env.GetClientArgus()
	var params = map[string]bool{"detail": true}

	Describe("功能测试", func() {
		Context("功能验证", func() {
			var params = map[string]bool{"detail": true}
			//功能性验证
			filelist := []string{
				"illegal_flag.jpg",
				"tibetan_flag.jpeg",
				"guns.jpg",
				"knives.jpg",
				"bloodiness.jpeg",
				"bomb_fire.jpg",
				"bomb.jpeg",
				"bomb_vehicle.png",
				"bomb_self_burning.jpeg",
				"march.jpeg",
				"march_crowed.jpg",
				"special_characters.jpg",
				"masked.jpeg",
				"fight_person.jpg",
				"fight_police.jpg",
				"anime_bloodiness.jpeg",
				"anime_likely_bomb.jpg",
				"islamic_dress.png",
				"normaltest.jpeg"}
			labellist := []int{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0}
			classlist := []string{
				"islamic flag",
				"tibetan flag",
				"guns",
				"knives",
				"bloodiness_human",
				"bomb_fire",
				"bomb_smoke",
				"bomb_vehicle",
				"bomb_self-burning",
				"march_banner",
				"march_crowed",
				"character",
				"masked",
				"fight_person",
				"fight_police",
				"anime_likely_bloodiness",
				"anime_likely_bomb",
				"islamic_dress",
				""}
			imgCh := make(chan string, len(filelist))
			imgI := make(chan int, len(filelist))
			for i, img := range filelist {
				var fileImg P.Imgfile
				fileImg.Imgname = img
				uri := E.Env.GetURIPrivate(imgSet + img)
				err := biz.StoreUri(imgSet+img, uri)
				fileImg.Imgurl = uri
				if err != nil {
					panic(err)
				}
				imgCh <- fileImg.Imgurl
				imgI <- i
				fmt.Println("下载图片" + img)
				It("测试图片："+fileImg.Imgname, func() {
					//调用api获取最新结果
					// file 总是pulpfiles的最后一个数据，故用fileName
					ii := <-imgI
					uri, ok := <-imgCh
					Expect(ok).To(Equal(true))
					resp, err := client.PostWithJson(path,
						proto.NewArgusReq(uri, nil, params))
					Expect(err).Should(BeNil())
					Expect(resp.Status()).To(Equal(200))
					//以下为验证结果
					var resObj Tcomplex.TerrorComplexResp
					Expect(json.Unmarshal(resp.BodyToByte(), &resObj)).Should(BeNil())
					Expect(resObj.Result.Label).Should(Equal(labellist[ii]))
					if classlist[ii] != "" {
						Expect(resObj.Result.Classes[0].Class).To(Equal(classlist[ii]))
						Expect(resObj.Result.Score).To(BeNumerically("~", resObj.Result.Classes[0].Score, 0.000001))
					}

				})
			}
			It("String Detail", func() {
				var StrParams = map[string]string{"detail": "true"}
				resp, err := client.PostWithJson(path,
					proto.NewArgusReq(E.Env.GetURIPrivate(imgSet+filelist[3]), nil, StrParams))
				Expect(err).Should(BeNil())
				Expect(resp.Status()).To(Equal(599))
				msg := resp.ResponseBodyAsJson()["error"].(string)
				Expect(msg).Should(Equal("json: cannot unmarshal string into Go struct field .detail of type bool"))
			})
		})
		Context("多分类", func() {
			terrorfiles := []string{
				"1_bk_bomb_fire_0523_00000007.jpg",
				"terror_complex.png",
				"20170930_isis_flag_499.jpg",
				"7_bk_march_banner_0523_00000002.jpg",
				"bk_more_guns.jpg",
				"tibetan_flag_march_banner.jpg",
				"army_guns.jpeg"}
			labellist := []int{1, 0, 1, 0, 1, 1, 1}
			classlist := [][]string{
				{"bomb_fire"},
				{},
				{"beheaded_isis", "isis flag", "guns"},
				{},
				{"guns"},
				{"tibetan flag", "march_banner"},
				{"army", "guns"}}
			imgCh := make(chan string, len(terrorfiles))
			imgI := make(chan int, len(terrorfiles))
			for i, img := range terrorfiles {
				var fileImg P.Imgfile
				fileImg.Imgname = img
				uri := E.Env.GetURIPrivate(imgSet + img)
				err := biz.StoreUri(imgSet+img, uri)
				fileImg.Imgurl = uri
				if err != nil {
					panic(err)
				}
				imgCh <- fileImg.Imgurl
				imgI <- i
				It("测试图片："+fileImg.Imgname, func() {
					// terrorfiles := []string{"1_bk_bomb_fire_0523_00000001.jpg", "1_bk_bomb_fire_0523_00000007.jpg"}
					ii := <-imgI
					uri, ok := <-imgCh
					Expect(ok).To(Equal(true))
					resp, err := client.PostWithJson(path,
						proto.NewArgusReq(uri, nil, params))
					Expect(err).Should(BeNil())
					Expect(resp.Status()).To(Equal(200))
					var resObj Tcomplex.TerrorComplexResp
					err = json.Unmarshal(resp.BodyToByte(), &resObj)
					Expect(err).Should(BeNil())
					if resObj.Result.Score < 0.83 {
						Expect(resObj.Result.Review).To(Equal(true))
					} else {
						Expect(resObj.Result.Review).To(Equal(false))
					}
					Expect(resObj.Result.Label).To(Equal(labellist[ii]))
					if labellist[ii] == 1 {
						Expect(len(resObj.Result.Classes)).To(BeNumerically(">", 0))
						Expect(resObj.Result.Score).To(BeNumerically("~", resObj.Result.Classes[0].Score, 0.000001))
						for j, class := range classlist[ii] {
							Expect(len(resObj.Result.Classes)).To(Equal(len(classlist[ii])))
							Expect(resObj.Result.Classes[j].Class).To(Equal(class))
						}
					}

				})
			}
		})
		Context("测试图片库", func() {
			imgList, err := P.BatchGetImg("normal")
			if err != nil {
				panic(err)
			}
			imgCh := make(chan string, len(imgList))
			for _, img := range imgList {
				err = biz.StoreUri(img.Imgname, img.Imgurl)
				if err != nil {
					panic(err)
				}
				imgCh <- img.Imgurl
				It("测试图片:"+img.Imgname, func() {
					url, ok := <-imgCh
					Expect(ok).To(Equal(true))
					resp, err := client.PostWithJson(path,
						proto.NewArgusReq(url, nil, params))
					Expect(err).Should(BeNil())
					Expect(resp.Status()).To(Or(Equal(200), BeNumerically("<", 600)))
					var resObj proto.ArgusRes
					err = json.Unmarshal(resp.BodyToByte(), &resObj)
					Expect(err).Should(BeNil())

				})
			}

		})
		Context("Check Code", func() {
			assert.CheckImageCode(client, path,nil)
		})
	})

})
