package test

import (
	"encoding/json"
	"time"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	EvalTest "qiniu.com/argus/service/service/image/terror/image_sync/test/evaltest"
	Test "qiniu.com/argus/service/service/video/censor/video/test"
	E "qiniu.com/argus/test/biz/env"
	"qiniu.com/argus/test/biz/proto"
	biz "qiniu.com/argus/test/biz/util"
	"qiniu.com/argus/test/configs"
)

var _ = Describe("[censor]|[video]|[terror]|v3/censor/video", func() {
	var (
		filePath   = "ccp/censor/"
		scene      = "terror"
		path       = biz.GetPath(scene, "video", "path")
		client     = E.Env.GetClientArgusVideo()
		normalFile = []string{"video-normal.mp4"}
		terrorFile = []string{"video-terror-review-fight-person.3gp", "video-terror-block-gun-2.3gp", "video-terror-block-gun.ts"}
	)
	normalUrls := Test.UploadVideoFile(filePath, normalFile)
	terrorUrls := Test.UploadVideoFile(filePath, terrorFile)
	Describe("Sync", func() {

		Context("v3|terror", func() {
			It("pass", func() {
				uri := normalUrls[0]
				req := proto.NewCensorVideoReq(uri, []string{"terror"}, 1000)
				resp, err := client.PostWithJson(path, req)
				Expect(err).Should(BeNil())
				Expect(resp.Status()).To(Equal(200))

				var tmp proto.VRespV3
				err = json.Unmarshal(resp.BodyToByte(), &tmp)
				Expect(err).Should(BeNil())
				result := tmp.Result
				Expect(result.Suggestion).To(Equal("pass"))
				Expect(result.Scenes.Terror.Suggestion).To(Equal("pass"))
				Expect(len(result.Scenes.Terror.Cuts)).Should(BeNumerically(">", 0))
				Expect(result.Scenes.Terror.Cuts[0].Suggestion).To(Equal("pass"))
				Test.CheckVideoDetailV3(result.Scenes.Terror.Cuts[0].Details, "terror", "normal", "pass", "")
			})
			It("review", func() {
				uri := terrorUrls[0]
				req := proto.NewCensorVideoReq(uri, []string{"terror"}, 1000)
				resp, err := client.PostWithJson(path, req)
				Expect(err).Should(BeNil())
				Expect(resp.Status()).To(Equal(200))

				var tmp proto.VRespV3
				err = json.Unmarshal(resp.BodyToByte(), &tmp)
				Expect(err).Should(BeNil())
				result := tmp.Result
				Expect(result.Suggestion).To(Equal("review"))
				Expect(result.Scenes.Terror.Suggestion).To(Equal("review"))
				Expect(len(result.Scenes.Terror.Cuts)).Should(BeNumerically(">", 0))
				Expect(result.Scenes.Terror.Cuts[6].Suggestion).To(Equal("review"))
				Test.CheckVideoDetailV3(result.Scenes.Terror.Cuts[6].Details, "terror", "fight_person", "review", "")
			})
			It("block", func() {
				uri := terrorUrls[1]
				req := proto.NewCensorVideoReq(uri, []string{"terror"}, 1000)
				resp, err := client.PostWithJson(path, req)
				Expect(err).Should(BeNil())
				Expect(resp.Status()).To(Equal(200))

				var tmp proto.VRespV3
				err = json.Unmarshal(resp.BodyToByte(), &tmp)
				Expect(err).Should(BeNil())
				result := tmp.Result
				Expect(result.Suggestion).To(Equal("block"))
				Expect(result.Scenes.Terror.Suggestion).To(Equal("block"))
				Expect(len(result.Scenes.Terror.Cuts)).Should(BeNumerically(">", 0))
				Expect(result.Scenes.Terror.Cuts[15].Suggestion).To(Equal("block"))
				Test.CheckVideoDetailV3(result.Scenes.Terror.Cuts[15].Details, "terror", "guns", "block", "")
			})
			It("block|ts文件", func() {
				uri := terrorUrls[2]
				req := proto.NewCensorVideoReq(uri, []string{"terror"}, 1000)
				resp, err := client.PostWithJson(path, req)
				Expect(err).Should(BeNil())
				Expect(resp.Status()).To(Equal(200))

				var tmp proto.VRespV3
				err = json.Unmarshal(resp.BodyToByte(), &tmp)
				Expect(err).Should(BeNil())
				result := tmp.Result
				Expect(result.Suggestion).To(Equal("block"))
				Expect(result.Scenes.Terror.Suggestion).To(Equal("block"))
				Expect(len(result.Scenes.Terror.Cuts)).Should(BeNumerically(">", 0))
				Expect(result.Scenes.Terror.Cuts[10].Suggestion).To(Equal("block"))
				Test.CheckVideoDetailV3(result.Scenes.Terror.Cuts[10].Details, "terror", "guns", "block", "")
			})
		})
	})
	Describe("ErrorCode", func() {
		Test.CheckVideoCensorCode(client, scene, path)
	})
	Describe("Check Normal Video", func() {
		Test.CheckDiffTypeVideoCensor(client, scene, path)
	})
	Describe("Check Vframe Censor Video", func() {
		Test.CheckCensorVideoVframe(client, scene, path)
	})
	Describe("Async", func() {
		if !configs.StubConfigs.Servers.VideoAsync {
			return
		}
		Context("v3|terror", func() {
			It("review", func() {
				uri := terrorUrls[0]
				req := proto.NewCensorVideoReq(uri, []string{"terror"}, 1000)
				resp, err := client.PostWithJson(path, req)
				Expect(err).Should(BeNil())
				Expect(resp.Status()).To(Equal(200))

				var job proto.VRespJob
				Expect(json.Unmarshal(resp.BodyToByte(), &job)).Should(BeNil())

				{
					path := "/v3/jobs/video/" + job.Job
					var result proto.VRespJobResult
					for i := 0; i < 20; i++ {
						resp, err := client.Get(path)
						Expect(err).Should(BeNil())
						Expect(resp.Status()).To(Equal(200))

						Expect(json.Unmarshal(resp.BodyToByte(), &result)).Should(BeNil())
						Expect(result.ID).To(Equal(job.Job))
						Expect(result.Request).To(Equal(req))
						if result.Status == "FINISHED" {
							break
						}
						time.Sleep(time.Second * 3)
					}
					Expect(result.Status).To(Equal("FINISHED"))
					rt := result.Result.Result
					Expect(rt.Suggestion).To(Equal("review"))
					Expect(rt.Scenes.Terror.Suggestion).To(Equal("review"))
					Expect(len(rt.Scenes.Terror.Cuts)).Should(BeNumerically(">", 0))
					Expect(rt.Scenes.Terror.Cuts[0].Suggestion).To(Equal("review"))
					Test.CheckVideoDetailV3(rt.Scenes.Terror.Cuts[0].Details, "terror", "fight_person", "review", "")
				}
			})
		})
	})
})

var _ = Describe("[eval]|[TerrorDetect]|/v1/eval/...censor.terror.video.evalTerrorDetect", func() {
	EvalTest.EvalTerrorDetectTest("TerrorDetect", "evalTerrorDetect", "video")
})

var _ = Describe("[eval]|[TerrorMixup]|/v1/eval/...censor.terror.video.evalTerrorMixup", func() {
	EvalTest.EvalTerrorMixupTest("TerrorMixup", "evalTerrorMixup", "video")
})
