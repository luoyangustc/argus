package test

import (
	"encoding/json"
	"time"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	EvalTest "qiniu.com/argus/service/service/image/pulp/image_sync/test/evaltest"
	Test "qiniu.com/argus/service/service/video/censor/video/test"
	E "qiniu.com/argus/test/biz/env"
	"qiniu.com/argus/test/biz/proto"
	biz "qiniu.com/argus/test/biz/util"
	"qiniu.com/argus/test/configs"
)

var _ = Describe("[censor]|[video]|[pulp]|v3/censor/video", func() {
	var (
		filePath   = "ccp/censor/"
		scene      = "pulp"
		path       = biz.GetPath(scene, "video", "path")
		client     = E.Env.GetClientArgusVideo()
		normalFile = []string{"video-normal.mp4"}
		pulpFile   = []string{"video-pulp-sexy-2.mp4", "video-pulp-pulp-2.mp4"}
	)
	normalUrls := Test.UploadVideoFile(filePath, normalFile)
	pulpUrls := Test.UploadVideoFile(filePath, pulpFile)
	Describe("Sync", func() {

		Context("v3|pulp", func() {
			It("pass", func() {
				uri := normalUrls[0]
				req := proto.NewCensorVideoReq(uri, []string{"pulp"}, 1000)
				resp, err := client.PostWithJson(path, req)
				Expect(err).Should(BeNil())
				Expect(resp.Status()).To(Equal(200))

				var tmp proto.VRespV3
				err = json.Unmarshal(resp.BodyToByte(), &tmp)
				Expect(err).Should(BeNil())
				result := tmp.Result
				Expect(result.Suggestion).To(Equal("pass"))
				Expect(result.Scenes.Pulp.Suggestion).To(Equal("pass"))
				Expect(len(result.Scenes.Pulp.Cuts)).Should(BeNumerically(">", 0))
				Test.CheckVideoDetailV3(result.Scenes.Pulp.Cuts[0].Details, "pulp", "normal", "pass", "")
			})
			It("sexy", func() {
				uri := pulpUrls[0]
				req := proto.NewCensorVideoReq(uri, []string{"pulp"}, 1000)
				resp, err := client.PostWithJson(path, req)
				Expect(err).Should(BeNil())
				Expect(resp.Status()).To(Equal(200))

				var tmp proto.VRespV3
				err = json.Unmarshal(resp.BodyToByte(), &tmp)
				Expect(err).Should(BeNil())
				result := tmp.Result
				Expect(result.Suggestion).To(Equal("review"))
				Expect(result.Scenes.Pulp.Suggestion).To(Equal("review"))
				Expect(len(result.Scenes.Pulp.Cuts)).Should(BeNumerically(">", 0))
				Expect(result.Scenes.Pulp.Cuts[0].Suggestion).To(Equal("review"))
				Test.CheckVideoDetailV3(result.Scenes.Pulp.Cuts[0].Details, "pulp", "sexy", "review", "")
			})
			It("block", func() {
				uri := pulpUrls[1]
				req := proto.NewCensorVideoReq(uri, []string{"pulp"}, 1000)
				resp, err := client.PostWithJson(path, req)
				Expect(err).Should(BeNil())
				Expect(resp.Status()).To(Equal(200))

				var tmp proto.VRespV3
				err = json.Unmarshal(resp.BodyToByte(), &tmp)
				Expect(err).Should(BeNil())
				result := tmp.Result
				Expect(result.Suggestion).To(Equal("block"))
				Expect(result.Scenes.Pulp.Suggestion).To(Equal("block"))
				Expect(len(result.Scenes.Pulp.Cuts)).Should(BeNumerically(">", 0))
				Expect(result.Scenes.Pulp.Cuts[6].Suggestion).To(Equal("block"))
				Test.CheckVideoDetailV3(result.Scenes.Pulp.Cuts[6].Details, "pulp", "pulp", "block", "")
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
		Context("v3|pulp", func() {
			It("pass", func() {
				uri := normalUrls[0]
				req := proto.NewCensorVideoReq(uri, []string{"pulp"}, 1000)
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
					Expect(rt.Suggestion).To(Equal("pass"))
					Expect(rt.Scenes.Pulp.Suggestion).To(Equal("pass"))
					Expect(len(rt.Scenes.Pulp.Cuts)).Should(BeNumerically(">", 0))
					Test.CheckVideoDetailV3(rt.Scenes.Pulp.Cuts[0].Details, "pulp", "normal", "pass", "")
				}
			})
		})
	})
})

var _ = Describe("[eval]|[Pulp]|/v1/eval/...cesnor.pulp.video.evalPulp", func() {
	EvalTest.EvalPulpTest("Pulp", "evalPulp", "video")
})

var _ = Describe("[eval]|[PulpFilter]|/v1/eval/...censor.pulp.video.evalPulpFilter", func() {
	EvalTest.EvalPulpFilterTest("PulpFilter", "evalPulpFilter", "video")
})
