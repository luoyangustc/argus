package test

import (
	"encoding/json"
	"fmt"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	EvalTest "qiniu.com/argus/service/service/image/politician/image_sync/test/evaltest"
	Test "qiniu.com/argus/service/service/video/vod/video/test"
	"qiniu.com/argus/test/biz/assert"
	E "qiniu.com/argus/test/biz/env"
	"qiniu.com/argus/test/biz/proto"
	biz "qiniu.com/argus/test/biz/util"
	"qiniu.com/argus/test/configs"
	"qiniu.com/argus/test/lib/qnhttp"
)

//https://github.com/qbox/ava/blob/dev/docs/Argus/Argus.Video.ops.md#detection
var _ = Describe("[argus]|[video][vod]|[politician]", func() {
	var client = E.Env.GetClientArgusVideo()
	var server = "politician"
	var op = biz.GetOp(server, "videovod")
	var jobpath = "/v1/jobs/video/"
	var path = biz.GetPath(server, "videovod", "path") + "/111"
	var req *proto.ArgusVideoRequest
	var resp *qnhttp.Response
	var err error
	videoName := []string{"politician_s.mp4", "test5miao.mp4", "pulp_all.mp4"}
	var url []string
	for _, file := range videoName {
		videoUrl := E.Env.GetURIVideo(file)
		err = biz.StoreUri("argusvideo/"+file, videoUrl)
		if err != nil {
			panic(err)
		}
		url = append(url, videoUrl)
	}
	BeforeEach(func() {
		req = proto.NewArgusVideoRequest(url[0], op)
		req.SetVframe(0, 0)
		req.SetIgnore(true, 0)
	})
	Describe("Sync detection", func() {
		JustBeforeEach(func() {
			resp, err = client.PostWithJson(path, req)
			if err != nil {
				panic(err)
			}
		})
		Context("Basic", func() {
			It("Normal", func() {
				var res map[string]proto.VideoOpResult
				Expect(json.Unmarshal(resp.BodyToByte(), &res)).To(BeNil())
				Expect(resp.Status()).To(Equal(200))
				Test.CheckRes(res)
				CheckNormalPoliticianResp(res[op], op)
			})
		})
		Context("Basic no politician", func() {
			BeforeEach(func() {
				req = proto.NewArgusVideoRequest(url[1], op)
				req.SetIgnore(true, 0)
			})
			It("Normal all", func() {
				var res map[string]proto.VideoOpResult
				Expect(json.Unmarshal(resp.BodyToByte(), &res)).Should(BeNil())
				Expect(resp.Status()).To(Equal(200))
				Expect(len(res[op].Segments)).To(Equal(0))
			})
		})
		Context("Basic all", func() {
			BeforeEach(func() {
				req = proto.NewArgusVideoRequest(url[2], op)
				//包含无类别帧
			})
			It("Normal all", func() {
				var res map[string]proto.VideoOpResult
				Expect(json.Unmarshal(resp.BodyToByte(), &res)).Should(BeNil())
				Expect(resp.Status()).To(Equal(200))
				// V.CheckRes(res)
				Expect(len(res[op].Labels)).Should(Equal(0))
				Expect(len(res[op].Segments)).Should(Equal(1))
				exScore := []float64{0.18437083}
				for i, lab := range res[op].Labels {
					Expect(lab.Score).Should(BeNumerically("~", exScore[i], 0.001))
				}
				Expect(res[op].Segments[0].Offset_begin).Should(Equal(0))
				Expect(res[op].Segments[0].Offset_end).Should(Equal(10000))
				Expect(len(res[op].Segments[0].Cuts)).Should(Equal(3))
				rslt_op := "detections"
				for _, cut := range res[op].Segments[0].Cuts {
					if cut.Offset == 10000 {
						s_result, err_1 := json.Marshal(cut.Result)
						if err_1 != nil {
							panic(err_1)
						}
						var results map[string][]PoliticianResult
						err_2 := json.Unmarshal(s_result, &results)
						if err_2 == nil {
							for _, result := range results[rslt_op] {
								Expect(result.BoundingBox.Score).Should(BeNumerically(">", 0.3))
								Expect(result.Value.Score).Should(BeNumerically(">", 0.1))

								assert.CheckPts(result.BoundingBox.Pts)
								Expect(result.Value.Name).Should(Equal(""))
							}
						} else {
							panic(err_2)
						}
					}
				}
			})
		})
	})
	Describe("Abnormal test", func() {
		Test.CheckVideoAbnormal(client, op, path, "")
	})
	Describe("Check Vframe", func() {
		JustBeforeEach(func() {
			resp, err = client.PostWithJson(path, req)
			if err != nil {
				panic(err)
			}
		})
		Context("set vframe 0,2", func() {
			var interval = 2
			BeforeEach(func() {
				req.SetVframe(0, interval)
			})
			It("Normal all", func() {
				var res map[string]proto.VideoOpResult
				Expect(json.Unmarshal(resp.BodyToByte(), &res)).Should(BeNil())
				Expect(resp.Status()).To(Equal(200))
				Expect(len(res[op].Segments)).To(Equal(2))
				Expect(res[op].Segments[0].Offset_begin).Should(BeNumerically(">", 0))
				Expect(res[op].Segments[0].Offset_end).Should(BeNumerically(">=", res[op].Segments[0].Offset_begin))
				var firstCut = res[op].Segments[0].Cuts[0].Offset
				var id = 0
				for _, segment := range res[op].Segments {
					for _, cut := range segment.Cuts {
						Expect(cut.Offset).Should(Equal(firstCut + id*interval*1000))
						id++
					}
				}
			})
		})
		Context("set vframe 1,0", func() {
			var interval = 5
			BeforeEach(func() {
				req.SetVframe(1, 0)
				//默认截帧 5.0s
			})
			It("Normal all", func() {
				var res map[string]proto.VideoOpResult
				Expect(json.Unmarshal(resp.BodyToByte(), &res)).Should(BeNil())
				Expect(resp.Status()).To(Equal(200))
				Expect(len(res[op].Segments)).To(Equal(2))
				Expect(res[op].Segments[0].Offset_begin).Should(BeNumerically(">", 0))
				Expect(res[op].Segments[0].Offset_end).Should(BeNumerically(">=", res[op].Segments[0].Offset_begin))
				var firstCut = res[op].Segments[0].Cuts[0].Offset
				var id = 0
				for _, segment := range res[op].Segments {
					for _, cut := range segment.Cuts {
						Expect(cut.Offset).Should(Equal(firstCut + id*interval*1000))
						id++
					}
				}
			})
		})
	})
	Describe("Async Politician Job", func() {
		if !configs.StubConfigs.Servers.VideoAsync {
			return
		}
		JustBeforeEach(func() {
			resp, err = client.PostWithJson(path, req)
			if err != nil {
				panic(err)
			}
		})
		var job_id string
		var job_vid = "politicianjob"
		Context("normal", func() {
			BeforeEach(func() {
				path = biz.GetPath(server, "videovod", "path") + "/" + job_vid
				req = proto.NewArgusVideoRequest(url[0], op)
				req.SetVframe(0, 0)
				req.SetAsync(true)
				fmt.Println(path)
			})
			It("Basic", func() {
				Expect(resp.Status()).To(Equal(200))
				job_id = resp.ResponseBodyAsJson()["job"].(string)
				Res := Test.CheckVideoAsync(client, op, job_id, job_vid, jobpath)
				CheckNormalPoliticianResp(Res[op].Result, op)
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
		Context("set label 1", func() {
			BeforeEach(func() {
				//2只选择该类别
				req.AddLabel("习近平", 2, 0.9, 0)
			})
			It("Normal", func() {
				var res map[string]proto.VideoOpResult
				Expect(json.Unmarshal(resp.BodyToByte(), &res)).Should(BeNil())
				Expect(resp.Status()).To(Equal(200))
				Expect(len(res[op].Segments)).To(Equal(0))
			})
		})
		Context("set label 2", func() {
			BeforeEach(func() {
				//1 忽略小于score的值
				req.AddLabel("习近平", 1, 0.9, 0)
			})
			It("Normal", func() {
				var res map[string]proto.VideoOpResult
				Expect(json.Unmarshal(resp.BodyToByte(), &res)).Should(BeNil())
				Expect(resp.Status()).To(Equal(200))
				for _, label := range res[op].Segments[0].Labels {
					if label.Label == "习近平" {
						Expect(label.Label).ShouldNot(Equal("习近平"))
					}
				}
			})
		})
		Context("test teminal", func() {
			BeforeEach(func() {
				req.SetVframe(0, 1)
				req.SetTerminateMode(0, 1)
				req.AddTerminateLabels(0, "习近平", 1)
			})
			It("Test Teminal", func() {
				var res map[string]proto.VideoOpResult
				Expect(json.Unmarshal(resp.BodyToByte(), &res)).Should(BeNil())
				Expect(resp.Status()).To(Equal(200))
				Expect(len(res)).Should(Equal(1))
				Expect(len(res[op].Segments[0].Cuts)).Should(Equal(1))
			})
		})
	})
	Describe("Normal", func() {
		Test.CheckDiffTypeVideo(client, op, path, "")
	})
})

var _ = Describe("[eval]|[politician]|/v1/eval/...vod.politician.video.evalPolitician", func() {
	EvalTest.EvalPoliticianTest("Politician", "evalPolitician", "videovod")
})

type PoliticianResult struct {
	BoundingBox struct {
		Pts   [][2]int `json:"pts"`
		Score float64  `json:"score"`
	} `json:"boundingBox"`
	Value struct {
		Score  float64 `json:"score"`
		Name   string  `json:"name"`
		Review bool    `json:"review"`
	} `json:"value"`
	Sample struct {
		Url string   `json:"url"`
		Pts [][2]int `json:"pts"`
	} `json:"sample"`
}

func CheckNormalPoliticianResp(res proto.VideoOpResult, op string) {
	Expect(len(res.Labels)).Should(Equal(2))
	Expect(len(res.Segments)).Should(Equal(1))
	exScore := []float64{0.82240295, 0.58138967}
	for i, lab := range res.Labels {
		Expect(lab.Score).Should(BeNumerically("~", exScore[i], 0.001))
	}
	Expect(res.Labels[0].Label).Should(Equal("习近平"))
	Expect(res.Segments[0].Offset_begin).Should(Equal(5003))
	Expect(res.Segments[0].Offset_end).Should(Equal(5003))
	Expect(len(res.Segments[0].Cuts)).Should(Equal(1))
	rslt_op := "detections"
	for _, cut := range res.Segments[0].Cuts {
		s_result, err_1 := json.Marshal(cut.Result)
		if err_1 != nil {
			panic(err_1)
		}
		var results map[string][]PoliticianResult
		err_2 := json.Unmarshal(s_result, &results)
		if err_2 == nil {
			for i, result := range results[rslt_op] {
				Expect(result.BoundingBox.Score).Should(BeNumerically(">", 0.3))
				Expect(result.Value.Score).Should(BeNumerically(">", 0.1))
				assert.CheckPts(result.BoundingBox.Pts)
				if i == 0 {
					Expect(result.Value.Name).Should(Equal("习近平"))
					Expect(result.Value.Review).Should(Equal(false))
					Expect(result.Sample.Url).Should(Equal("http://peps.ai.qiniuapi.com/img-ed89ee5ee6a6a5fa47ef60ed9319645b.jpg"))
					assert.CheckPts(result.Sample.Pts)
				} else if i == 1 {
					Expect(result.Value.Name).Should(Equal("沈跃跃"))
				} else if i == 5 {
					Expect(result.Value.Name).Should(Equal("吉炳轩"))
				}
			}
		} else {
			panic(err_2)
		}
	}
}
