package test

import (
	"encoding/json"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"qiniu.com/argus/test/biz/assert"
	P "qiniu.com/argus/test/biz/batch"
	C "qiniu.com/argus/test/biz/client"
	E "qiniu.com/argus/test/biz/env"
	"qiniu.com/argus/test/biz/proto"
	biz "qiniu.com/argus/test/biz/util"
	"qiniu.com/argus/test/lib/qnhttp"
)

// CheckVideoDetailV3 ...
func CheckVideoDetailV3(lbDetail []*proto.CutDetail, scene string, label string, suggestion string, group string) {
	if len(lbDetail) == 0 {
		return
	}
	// 分类标签（不带pts）只选取最高分的一个，不同时展示多个;
	// 检测标签（带pts）按label合并，pts与score罗列在detections中
	classCnt := 0
	detectMap := make(map[string]int)
	bExist := false

	for _, detail := range lbDetail {
		if detail.Label == label {
			Expect(detail.Suggestion).To(Equal(suggestion))
			if scene == "politician" {
				Expect(detail.Group).To(Equal(group))
			}
			bExist = true
		}

		if len(detail.Detections) == 0 {
			classCnt++
			Expect(classCnt).Should(Equal(1))
		} else {
			detectMap[detail.Label]++
			Expect(detectMap[detail.Label]).Should(Equal(1))

			for _, d := range detail.Detections {
				Expect(d.Score).Should(BeNumerically("<=", detail.Score))
				// 验证坐标合法性
				assert.CheckPts(d.Pts)
			}
		}
	}
	Expect(bExist).To(Equal(true))
}

func UploadVideoFile(fileSet string, fileNames []string) []string {
	var fileUrls []string
	for _, file := range fileNames {
		fileName := file
		uri := E.Env.GetURIVideo(fileSet + fileName)
		if err := biz.StoreUri("argusvideo/"+fileSet+fileName, uri); err != nil {
			panic(err)
		}
		fileUrls = append(fileUrls, uri)
	}
	return fileUrls
}

func CheckDiffTypeVideoCensor(c C.Client, Scene string, Path string) {
	videoList, err := P.BatchGetVideo("AllTypeVideo/")
	if err != nil {
		panic(err)
	}
	Context("normal video", func() {
		for _, file := range videoList {
			var fileName = file
			By("video download:" + fileName.Imgname)
			if err := biz.StoreUri("argusvideo/"+fileName.Imgname, fileName.Imgurl); err != nil {
				panic(err)
			}
			It("测试视频"+fileName.Imgname, func() {
				req := proto.NewCensorVideoReq(fileName.Imgurl, []string{Scene}, 1000)
				resp, err := c.PostWithJson(Path, req)
				Expect(err).Should(BeNil())
				Expect(resp.Status()).To(Or((Equal(200)), (Equal(400))))
			})
		}
	})
}

func CheckVideoCensorCode(c C.Client, Scene string, Path string) {
	var (
		res  proto.CodeErrorResp
		req  *proto.VReqV3
		resp *qnhttp.Response
		err  error
	)
	var videoName = "test5miao.mp4"
	videourl := E.Env.GetURIVideo(videoName)
	err = biz.StoreUri("argusvideo/"+videoName, videourl)
	if err != nil {
		panic(err)
	}
	JustBeforeEach(func() {
		resp, err = c.PostWithJson(Path, req)
		if err != nil {
			panic(err)
		}
	})
	Context("url not exist", func() {
		BeforeEach(func() {
			req = proto.NewCensorVideoReq("not_exist.mp4", []string{Scene}, 1000)
		})
		It("ex", func() {
			Expect(resp.Status()).To(Equal(400))
			Expect(json.Unmarshal(resp.BodyToByte(), &res)).Should(BeNil())
			Expect(res.Code).Should(Equal(4000203))
			Expect(res.Error).Should(Equal("fetch uri failed"))
		})
	})
	Context("fetch uri failed", func() {
		var videoName = "bad.mp4"
		url := E.Env.GetURIVideo(videoName)
		err = biz.StoreUri("argusvideo/"+videoName, url)
		if err != nil {
			panic(err)
		}
		BeforeEach(func() {
			req = proto.NewCensorVideoReq(url, []string{Scene}, 1000)
		})
		It("ex", func() {
			Expect(resp.Status()).To(Equal(400))
			Expect(json.Unmarshal(resp.BodyToByte(), &res)).Should(BeNil())
			Expect(res.Code).Should(Equal(4000203))
			Expect(res.Error).Should(Equal("fetch uri failed"))
		})
	})
	Context("cannot find video", func() {
		var videoName = "audio-mp3.mp3"
		url := E.Env.GetURIVideo(videoName)
		err = biz.StoreUri("argusvideo/"+videoName, url)
		if err != nil {
			panic(err)
		}
		BeforeEach(func() {
			req = proto.NewCensorVideoReq(url, []string{Scene}, 1000)
		})
		It("ex", func() {
			Expect(resp.Status()).To(Equal(415))
			Expect(json.Unmarshal(resp.BodyToByte(), &res)).Should(BeNil())
			Expect(res.Code).Should(Equal(4150501))
			Expect(res.Error).Should(Equal("cannot find video"))
		})
	})
	Context("bad scene", func() {
		BeforeEach(func() {
			scene := "s"
			req = proto.NewCensorVideoReq(videourl, []string{scene}, 1000)
		})
		It("ex", func() {
			Expect(resp.Status()).To(Equal(400))
			Expect(json.Unmarshal(resp.BodyToByte(), &res)).Should(BeNil())
			Expect(res.Code).Should(Equal(4000100))
			Expect(res.Error).Should(Equal("bad scene"))
		})
	})
	Context("no scene", func() {
		BeforeEach(func() {
			scene := ""
			req = proto.NewCensorVideoReq(videourl, []string{scene}, 1000)
		})
		It("ex", func() {
			Expect(resp.Status()).To(Equal(400))
			Expect(json.Unmarshal(resp.BodyToByte(), &res)).Should(BeNil())
			Expect(res.Code).Should(Equal(4000100))
			Expect(res.Error).Should(Equal("bad scene"))
		})
	})
	Context("vframe mode bad", func() {
		BeforeEach(func() {
			req = proto.NewCensorVideoReq(videourl, []string{Scene}, 900)
		})
		It("ex", func() {
			Expect(resp.Status()).To(Equal(400))
			Expect(json.Unmarshal(resp.BodyToByte(), &res)).Should(BeNil())
			Expect(res.Code).Should(Equal(4000100))
			Expect(res.Error).Should(Equal("invalid interval, allow interval is [1000, 60000]"))
		})
	})
}

func CheckCensorVideoVframe(c C.Client, Scene string, Path string) {
	var (
		res  proto.VRespV3
		req  *proto.VReqV3
		resp *qnhttp.Response
		err  error
	)
	var videoName = "normal_vframe.mp4"
	videourl := E.Env.GetURIVideo(videoName)
	err = biz.StoreUri("argusvideo/"+videoName, videourl)
	if err != nil {
		panic(err)
	}
	JustBeforeEach(func() {
		resp, err = c.PostWithJson(Path, req)
		if err != nil {
			panic(err)
		}
	})
	//interval 取值范围1000～60000 单位：毫秒
	Context("vframe interval 1000ms", func() {
		BeforeEach(func() {
			req = proto.NewCensorVideoReq(videourl, []string{Scene}, 1000)
		})
		It("video: "+videoName, func() {
			Expect(resp.Status()).To(Equal(200))
			Expect(json.Unmarshal(resp.BodyToByte(), &res)).Should(BeNil())
			switch Scene {
			case "pulp":
				Expect(len(res.Result.Scenes.Pulp.Cuts)).To(Equal(12))
			case "poltician":
				Expect(len(res.Result.Scenes.Politician.Cuts)).To(Equal(12))
			case "terror":
				Expect(len(res.Result.Scenes.Terror.Cuts)).To(Equal(12))
			}
		})
	})
	Context("vframe interval 5000ms", func() {
		BeforeEach(func() {
			req = proto.NewCensorVideoReq(videourl, []string{Scene}, 5000)
		})
		It("video: "+videoName, func() {
			Expect(resp.Status()).To(Equal(200))
			Expect(json.Unmarshal(resp.BodyToByte(), &res)).Should(BeNil())
			switch Scene {
			case "pulp":
				Expect(len(res.Result.Scenes.Pulp.Cuts)).To(Equal(3))
			case "poltician":
				Expect(len(res.Result.Scenes.Politician.Cuts)).To(Equal(3))
			case "terror":
				Expect(len(res.Result.Scenes.Terror.Cuts)).To(Equal(3))
			}
		})
	})
	Context("vframe interval 60000ms", func() {
		BeforeEach(func() {
			req = proto.NewCensorVideoReq(videourl, []string{Scene}, 60000)
		})
		It("video: "+videoName, func() {
			Expect(resp.Status()).To(Equal(200))
			Expect(json.Unmarshal(resp.BodyToByte(), &res)).Should(BeNil())
			switch Scene {
			case "pulp":
				Expect(len(res.Result.Scenes.Pulp.Cuts)).To(Equal(1))
			case "poltician":
				Expect(len(res.Result.Scenes.Politician.Cuts)).To(Equal(1))
			case "terror":
				Expect(len(res.Result.Scenes.Terror.Cuts)).To(Equal(1))
			}
		})
	})
}
