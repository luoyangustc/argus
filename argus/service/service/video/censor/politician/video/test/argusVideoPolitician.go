package test

import (
	"encoding/json"
	"time"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	EvalTest "qiniu.com/argus/service/service/image/politician/image_sync/test/evaltest"
	Test "qiniu.com/argus/service/service/video/censor/video/test"
	E "qiniu.com/argus/test/biz/env"
	"qiniu.com/argus/test/biz/proto"
	biz "qiniu.com/argus/test/biz/util"
	"qiniu.com/argus/test/configs"
)

var _ = Describe("[censor]|[video]|[politician]|v3/censor/video", func() {
	var (
		filePath       = "ccp/censor/"
		scene          = "politician"
		path           = biz.GetPath(scene, "video", "path")
		client         = E.Env.GetClientArgusVideo()
		normalFile     = []string{"video-normal.mp4"}
		politicianFile = []string{"video-politician-review.3gp", "video-politician-block.mp4"}
	)
	normalUrls := Test.UploadVideoFile(filePath, normalFile)
	politicianUrls := Test.UploadVideoFile(filePath, politicianFile)
	Describe("Sync", func() {

		Context("v3|politician", func() {
			It("pass", func() {
				uri := normalUrls[0]
				req := proto.NewCensorVideoReq(uri, []string{"politician"}, 1000)
				resp, err := client.PostWithJson(path, req)
				Expect(err).Should(BeNil())
				Expect(resp.Status()).To(Equal(200))

				var tmp proto.VRespV3
				err = json.Unmarshal(resp.BodyToByte(), &tmp)
				Expect(err).Should(BeNil())
				result := tmp.Result
				Expect(result.Suggestion).To(Equal("pass"))
				Expect(result.Scenes.Politician.Suggestion).To(Equal("pass"))
				Expect(len(result.Scenes.Politician.Cuts)).Should(BeNumerically(">", 0))
				Test.CheckVideoDetailV3(result.Scenes.Politician.Cuts[0].Details, "politician", "normal", "pass", "")
			})
			It("review", func() {
				uri := politicianUrls[0]
				req := proto.NewCensorVideoReq(uri, []string{"politician"}, 1000)
				resp, err := client.PostWithJson(path, req)
				Expect(err).Should(BeNil())
				Expect(resp.Status()).To(Equal(200))

				var tmp proto.VRespV3
				err = json.Unmarshal(resp.BodyToByte(), &tmp)
				Expect(err).Should(BeNil())
				result := tmp.Result
				Expect(result.Suggestion).To(Equal("review"))
				Expect(result.Scenes.Politician.Suggestion).To(Equal("review"))
				Expect(len(result.Scenes.Politician.Cuts)).Should(BeNumerically(">", 0))
				Expect(result.Scenes.Politician.Cuts[0].Suggestion).To(Equal("review"))
				Test.CheckVideoDetailV3(result.Scenes.Politician.Cuts[0].Details, "politician", "习近平", "review", "domestic_statesman")
			})
			It("block", func() {
				uri := politicianUrls[1]
				req := proto.NewCensorVideoReq(uri, []string{"politician"}, 1000)
				resp, err := client.PostWithJson(path, req)
				Expect(err).Should(BeNil())
				Expect(resp.Status()).To(Equal(200))

				var tmp proto.VRespV3
				err = json.Unmarshal(resp.BodyToByte(), &tmp)
				Expect(err).Should(BeNil())
				result := tmp.Result
				Expect(result.Suggestion).To(Equal("block"))
				Expect(result.Scenes.Politician.Suggestion).To(Equal("block"))
				Expect(len(result.Scenes.Politician.Cuts)).Should(BeNumerically(">", 0))
				Expect(result.Scenes.Politician.Cuts[2].Suggestion).To(Equal("block"))
				Test.CheckVideoDetailV3(result.Scenes.Politician.Cuts[2].Details, "politician", "柴玲", "block", "anti_china_people")
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
		Context("v3|politician", func() {
			It("block", func() {
				uri := politicianUrls[1]
				req := proto.NewCensorVideoReq(uri, []string{"politician"}, 1000)
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
					Expect(rt.Suggestion).To(Equal("block"))
					Expect(rt.Scenes.Politician.Suggestion).To(Equal("block"))
					Expect(len(rt.Scenes.Politician.Cuts)).Should(BeNumerically(">", 0))
					Expect(rt.Scenes.Politician.Cuts[2].Suggestion).To(Equal("block"))
					Test.CheckVideoDetailV3(rt.Scenes.Politician.Cuts[2].Details, "politician", "柴玲", "block", "anti_china_people")
				}
			})
		})
	})
})

var _ = Describe("[eval]|[politician]|/v1/eval/...censor.politician.video.evalPolitician", func() {
	EvalTest.EvalPoliticianTest("Politician", "evalPolitician", "video")
})
