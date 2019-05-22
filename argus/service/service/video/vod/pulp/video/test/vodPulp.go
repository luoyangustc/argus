package test

import (
	"encoding/json"
	"fmt"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	EvalTest "qiniu.com/argus/service/service/image/pulp/image_sync/test/evaltest"
	Test "qiniu.com/argus/service/service/video/vod/video/test"
	E "qiniu.com/argus/test/biz/env"
	"qiniu.com/argus/test/biz/proto"
	biz "qiniu.com/argus/test/biz/util"
	"qiniu.com/argus/test/configs"
	"qiniu.com/argus/test/lib/qnhttp"
)

var _ = Describe("[argus]|[video]|[vod]|[pulp]", func() {
	var client = E.Env.GetClientArgusVideo()
	var server = "pulp"
	var op = biz.GetOp(server, "videovod")
	var jobpath = "/v1/jobs/video/"
	var path = biz.GetPath(server, "videovod", "path") + "/111"
	var req *proto.ArgusVideoRequest
	var resp *qnhttp.Response
	var err error
	var videoName = "pulp_all.mp4"
	url := E.Env.GetURIVideo(videoName)
	err = biz.StoreUri("argusvideo/"+videoName, url)
	if err != nil {
		panic(err)
	}
	BeforeEach(func() {
		req = proto.NewArgusVideoRequest(url, op)
		req.SetVframe(0, 0)
	})
	Describe("Sync Pulp", func() {
		JustBeforeEach(func() {
			resp, err = client.PostWithJson(path, req)
			if err != nil {
				panic(err)
			}
		})
		Context("Basic", func() {
			It("Normal", func() {
				var res map[string]proto.VideoOpResult
				Expect(json.Unmarshal(resp.BodyToByte(), &res)).Should(BeNil())
				Expect(resp.Status()).To(Equal(200))
				Test.CheckRes(res)
				Expect(len(res[op].Labels)).Should(Equal(2))
				Expect(len(res[op].Segments)).Should(Equal(2))
				for _, lab := range res[op].Labels {
					Expect(lab.Score).Should(BeNumerically("~", 0.99, 0.01))
				}
				Expect(res[op].Labels[0].Label).Should(Equal("2"))
				Expect(res[op].Labels[1].Label).Should(Equal("0"))
				Expect(res[op].Segments[0].Offset_begin).Should(Equal(0))
				Expect(res[op].Segments[0].Offset_end).Should(Equal(5000))
				Expect(len(res[op].Segments[0].Cuts)).Should(Equal(2))
				ex_label := [3]int{2, 2, 0}
				for i, cut := range res[op].Segments[0].Cuts {
					s_result, err_1 := json.Marshal(cut.Result)
					if err_1 != nil {
						panic(err_1)
					}
					var result PulpResult
					err_2 := json.Unmarshal(s_result, &result)
					//result, ok := cut.Result.(PulpResult)
					if err_2 == nil {
						Expect(result.Score).Should(BeNumerically(">", 0.6))
						Expect(result.Review).Should(Equal(false))
						Expect(result.Label).Should(Equal(ex_label[i]))
					} else {
						panic(err_2)
					}
				}
			})
		})
	})
	Describe("Abnormal test", func() {
		Test.CheckVideoAbnormal(client, op, path, "")
	})
	Describe("Check Vframe", func() {
		Test.CheckVideoSetVframe(client, op, path)
	})
	Describe("Async Pulp Job", func() {
		if !configs.StubConfigs.Servers.VideoAsync {
			return
		}
		var job_id string
		var job_vid = "pulpjob"
		JustBeforeEach(func() {
			resp, err = client.PostWithJson(path, req)
			if err != nil {
				panic(err)
			}
		})
		Context("normal", func() {
			BeforeEach(func() {
				path = biz.GetPath(server, "videovod", "path") + "/" + job_vid
				req = proto.NewArgusVideoRequest(url, op)
				req.SetVframe(0, 3)
				req.SetAsync(true)
				fmt.Println(path)
			})
			It("Basic", func() {
				Expect(resp.Status()).To(Equal(200))
				job_id = resp.ResponseBodyAsJson()["job"].(string)
				Result := Test.CheckVideoAsync(client, op, job_id, job_vid, jobpath)
				for s_op, res := range Result {
					Expect(s_op).Should(Equal(op))
					Expect(res.Code).Should(Equal(0))
					res_op := res.Result
					Expect(len(res_op.Labels)).Should(Equal(len(res_op.Segments)))
					for _, label := range res_op.Labels {
						Expect(label.Label).ShouldNot(Equal(""))
						Expect(label.Score).Should(BeNumerically(">", 0.1))
					}
					for i, segment := range res_op.Segments {
						Expect(segment.Offset_begin).Should(BeAssignableToTypeOf(0))
						Expect(segment.Offset_end).Should(BeAssignableToTypeOf(0))
						Expect(len(segment.Labels)).Should(Equal(1))
						for _, labels := range segment.Labels {
							Expect(labels.Label).Should(Equal(res_op.Labels[i].Label))
							Expect(labels.Score).Should(Equal(res_op.Labels[i].Score))
						}
						Expect(len(segment.Cuts)).Should(BeNumerically(">=", 1))
						for _, cut := range segment.Cuts {
							Expect(cut.Offset).Should(BeAssignableToTypeOf(0))
							cutResult, ok := cut.Result.(map[string]interface{})
							Expect(ok).Should(Equal(true))
							Expect(cutResult["label"]).ShouldNot(BeNil())
							Expect(cutResult["label"]).ShouldNot(Equal(""))
							Expect(cutResult["score"]).ShouldNot(BeNil())
							Expect(cutResult["score"]).Should(BeNumerically(">", 0.1))
						}
					}
				}
			})
		})
	})
	Describe("Check label", func() {
		JustBeforeEach(func() {
			resp, err = client.PostWithJson(path, req)
			if err != nil {
				panic(err)
			}
		})
		Context("test label 1", func() {
			BeforeEach(func() {
				//2只选择该类别
				req.AddLabel("2", 2, 0.7, 0)
			})
			It("Normal", func() {
				var res map[string]proto.VideoOpResult
				Expect(json.Unmarshal(resp.BodyToByte(), &res)).Should(BeNil())
				Expect(resp.Status()).To(Equal(200))
				Test.CheckRes(res)
				Expect(len(res)).Should(Equal(1))
				Expect(len(res[op].Labels)).Should(Equal(1))
				//Expect(len(res[op].Segments)).Should(Equal(1))
				Expect(len(res[op].Segments[0].Cuts)).Should(Equal(1))
			})
		})
		Context("test label 2", func() {
			BeforeEach(func() {
				//1忽略该类别
				req.AddLabel("2", 1, 0.7, 0)
			})
			It("Normal", func() {
				var res map[string]proto.VideoOpResult
				Expect(json.Unmarshal(resp.BodyToByte(), &res)).Should(BeNil())
				Expect(resp.Status()).To(Equal(200))
				Test.CheckRes(res)
				Expect(len(res)).Should(Equal(1))
				Expect(len(res[op].Labels)).Should(Equal(2))
				Expect(len(res[op].Segments[0].Cuts)).Should(Equal(1))
				Expect(len(res[op].Segments[0].Cuts)).Should(Equal(1))
			})
		})
		Context("test label 3", func() {
			BeforeEach(func() {
				//2选择该类别
				req.AddLabel("0", 2, 0.7, 0)
				req.AddLabel("2", 2, 0.7, 0)
			})
			It("multi label and", func() {
				var res map[string]proto.VideoOpResult
				Expect(json.Unmarshal(resp.BodyToByte(), &res)).Should(BeNil())
				Expect(resp.Status()).To(Equal(200))
				Test.CheckRes(res)
				Expect(len(res)).Should(Equal(1))
				Expect(len(res[op].Labels)).Should(Equal(2))
				//Expect(len(res[op].Segments)).Should(Equal(1))
				Expect(len(res[op].Segments[0].Cuts)).Should(Equal(1))
			})
		})
		Context("test label 4", func() {
			BeforeEach(func() {
				//1忽略该类别
				req.AddLabel("0", 1, 0.7, 0)
				req.AddLabel("2", 1, 0.7, 0)
			})
			It("multi label or", func() {
				var res map[string]proto.VideoOpResult
				Expect(json.Unmarshal(resp.BodyToByte(), &res)).Should(BeNil())
				Expect(resp.Status()).To(Equal(200))
				Test.CheckRes(res)
				Expect(len(res)).Should(Equal(1))
				Expect(len(res[op].Labels)).Should(Equal(2))
				//Expect(len(res[op].Segments)).Should(Equal(1))
				Expect(len(res[op].Segments[0].Cuts)).Should(Equal(1))
			})
		})
		Context("test label 5", func() {
			BeforeEach(func() {
				//1忽略该类别
				req.AddLabel("2", 1, 0, 0)
			})
			It("when label.select == 1, then label.score = 1.0 by default", func() {
				var res map[string]proto.VideoOpResult
				Expect(json.Unmarshal(resp.BodyToByte(), &res)).Should(BeNil())
				Expect(resp.Status()).To(Equal(200))
				Test.CheckRes(res)
				Expect(len(res)).Should(Equal(1))
				Expect(len(res[op].Labels)).Should(Equal(1))
				Expect(len(res[op].Segments[0].Cuts)).Should(Equal(1))
			})
		})
		Context("no result", func() {
			var videoName = "test5miao.mp4"
			url := E.Env.GetURIVideo(videoName)
			err = biz.StoreUri("argusvideo/"+videoName, url)
			if err != nil {
				panic(err)
			}
			BeforeEach(func() {
				req = proto.NewArgusVideoRequest(url, op)
				req.AddLabel("0", 2, 0.7, 0)
			})
			It("ex", func() {
				Expect(resp.Status()).To(Equal(200))
				var res map[string]proto.VideoOpResult
				Expect(json.Unmarshal(resp.BodyToByte(), &res)).Should(BeNil())
				Expect(len(res[op].Segments)).Should(Equal(0))
			})
		})
		Context("test teminal", func() {
			BeforeEach(func() {
				req.SetVframe(0, 1)
				req.SetTerminateMode(0, 1)
				req.AddTerminateLabels(0, "0", 2)
				req.AddTerminateLabels(0, "2", 2)
			})
			It("Test Teminal", func() {
				var res map[string]proto.VideoOpResult
				Expect(json.Unmarshal(resp.BodyToByte(), &res)).Should(BeNil())
				Expect(resp.Status()).To(Equal(200))
				Test.CheckRes(res)
				Expect(len(res)).Should(Equal(1))
				Expect(len(res[op].Labels)).Should(Equal(1))
				Expect(len(res[op].Labels)).Should(Equal(len(res[op].Segments[0].Labels)))
				Expect(len(res[op].Segments[0].Cuts)).Should(Equal(2))
			})
		})
	})
	Describe("Normal", func() {
		Test.CheckDiffTypeVideo(client, op, path, "")
	})
})

var _ = Describe("[eval]|[Pulp]|/v1/eval/...vod.pulp.video.evalPulp", func() {
	EvalTest.EvalPulpTest("Pulp", "evalPulp", "videovod")
})

var _ = Describe("[eval]|[PulpFilter]|/v1/eval/...vod.pulp.video.evalPulpFilter", func() {
	EvalTest.EvalPulpFilterTest("PulpFilter", "evalPulpFilter", "videovod")
})

type PulpResult struct {
	Label   int     `json:"label"`
	Score   float64 `json:"score"`
	Review  bool    `json:"review"`
	Class   string  `json:"class,omitempty"`
	Classes []struct {
		Class string  `json:"class,omitempty"`
		Score float64 `json:"score"`
	} `json:"classes,omitempty"`
}
