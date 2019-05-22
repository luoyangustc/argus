package test

import (
	"encoding/json"
	"fmt"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	Test "qiniu.com/argus/service/service/video/vod/video/test"
	E "qiniu.com/argus/test/biz/env"
	"qiniu.com/argus/test/biz/proto"
	biz "qiniu.com/argus/test/biz/util"
	"qiniu.com/argus/test/configs"
	"qiniu.com/argus/test/lib/qnhttp"
)

var _ = Describe("[argus][video]|[terror-complex]", func() {
	var client = E.Env.GetClientArgusVideo()
	var server = "terrorcomplex"
	var op = biz.GetOp(server, "videovod")
	var jobpath = "/v1/jobs/video/"
	var path = biz.GetPath(server, "videovod", "path") + "/111"
	var req *proto.ArgusVideoRequest
	var resp *qnhttp.Response
	var err error
	var videoName = "terror_s.mp4"
	url := E.Env.GetURIVideo(videoName)
	err = biz.StoreUri("argusvideo/"+videoName, url)
	if err != nil {
		panic(err)
	}
	BeforeEach(func() {
		req = proto.NewArgusVideoRequest(url, op)
		req.SetVframe(0, 0)
	})

	Describe("Sync terror-complex", func() {
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
				CheckNormalTerrorComplexResp(res[op], op)
			})

		})
		Context("detail", func() {
			BeforeEach(func() {
				req.SetDetail(0, true)
			})
			It("Normal", func() {
				var res map[string]proto.VideoOpResult
				Expect(json.Unmarshal(resp.BodyToByte(), &res)).To(BeNil())
				Expect(resp.Status()).To(Equal(200))
				Expect(len(res[op].Labels)).Should(Equal(2))
				Expect(len(res[op].Segments)).Should(Equal(3))
				ex_label_score := [2]float64{0.999572, 0.9977869}
				ex_label_label := [2]string{"0", "1"}
				Expect(len(res[op].Labels)).Should(Equal(len(ex_label_label)))
				for i, lab := range res[op].Labels {
					Expect(lab.Score).Should(BeNumerically("~", ex_label_score[i], 0.01))
					Expect(lab.Label).Should(Equal(ex_label_label[i]))
				}
				Expect(res[op].Segments[0].Offset_begin).Should(Equal(200))
				Expect(res[op].Segments[0].Offset_end).Should(Equal(10000))
				Expect(len(res[op].Segments[0].Cuts)).Should(Equal(3))

				ex_label := [3][]int{{0, 0, 0}, {1}, {0, 0}}
				ex_score := [3][]float64{{0.9793467, 0.69710433, .999572}, {0.9977869}, {0.9905856, 0.9552808}}
				ex_classes := [3][][]string{{}, {{"guns"}}, {}}

				for i, seg := range res[op].Segments {
					for j, cut := range seg.Cuts {
						s_result, err_1 := json.Marshal(cut.Result)
						if err_1 != nil {
							panic(err_1)
						}
						var result TerrorComplexResult
						err_2 := json.Unmarshal(s_result, &result)
						Expect(err_2).Should(BeNil())
						Expect(result.Score).Should(BeNumerically("~", ex_score[i][j], 0.001))
						Expect(result.Label).Should(Equal(ex_label[i][j]))
						if ex_label[i][j] == 0 {
							continue
						}
						Expect(len(result.Classes)).Should(Equal(len(ex_classes[i][j])))
						for k, class := range result.Classes {
							Expect(class.Class).Should(Equal(ex_classes[i][j][k]))
						}
					}
				}
			})

		})
		Context("detail string", func() {
			BeforeEach(func() {
				req.SetDetailString(0, "true")
			})
			It("Normal", func() {
				Expect(resp.Status()).To(Equal(400))
				msg := resp.ResponseBodyAsJson()["error"].(string)
				Expect(msg).Should(Equal("json: cannot unmarshal string into Go struct field .detail of type bool"))
			})
		})
	})
	Describe("Abnormal test", func() {
		Test.CheckVideoAbnormal(client, op, path, "")
	})
	Describe("Check Vframe", func() {
		Test.CheckVideoSetVframe(client, op, path)
	})
	Describe("Async Terror Complex", func() {
		if !configs.StubConfigs.Servers.VideoAsync {
			return
		}
		var job_id string
		var job_vid = "terror-complexjob"
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
				Res := Test.CheckVideoAsync(client, op, job_id, job_vid, jobpath)
				CheckNormalTerrorComplexResp(Res[op].Result, op)
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
				req.AddLabel("0", 2, 1.0, 0)
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
				//1	忽略小于score的值
				req.AddLabel("0", 1, 0.8, 0)
			})
			It("Normal", func() {
				var res map[string]proto.VideoOpResult
				Expect(json.Unmarshal(resp.BodyToByte(), &res)).Should(BeNil())
				Expect(resp.Status()).To(Equal(200))
				for _, segment := range res[op].Segments {
					for _, cut := range segment.Cuts {
						s_result, err_1 := json.Marshal(cut.Result)
						if err_1 != nil {
							panic(err_1)
						}
						var Res TerrorComplexResult
						Expect(json.Unmarshal(s_result, &Res)).Should(BeNil())
						if Res.Label == 0 {
							Expect(Res.Score).Should(BeNumerically(">=", 0.8))
						}
					}
				}
			})
		})
	})
	Describe("Normal", func() {
		Test.CheckDiffTypeVideo(client, op, path, "")
	})

})

type TerrorComplexResult struct {
	Label   int     `json:"label"`
	Score   float64 `json:"score"`
	Review  bool    `json:"review"`
	Class   string  `json:"class,omitempty"`
	Classes []struct {
		Class string  `json:"class,omitempty"`
		Score float64 `json:"score"`
	} `json:"classes,omitempty"`
}

func CheckNormalTerrorComplexResp(res proto.VideoOpResult, op string) {
	Expect(len(res.Labels)).Should(Equal(2))
	Expect(len(res.Segments)).Should(Equal(3))
	ex_label_score := [2]float64{0.999572, 0.9977869}
	for i, lab := range res.Labels {
		Expect(lab.Score).Should(BeNumerically("~", ex_label_score[i], 0.01))
	}
	Expect(res.Labels[0].Label).Should(Equal("0"))
	Expect(res.Labels[1].Label).Should(Equal("1"))
	Expect(res.Segments[0].Offset_begin).Should(Equal(200))
	Expect(res.Segments[0].Offset_end).Should(Equal(10000))
	Expect(len(res.Segments[0].Cuts)).Should(Equal(3))
	ex_label := [3][]int{{0, 0, 0}, {1}, {0, 0}}
	ex_score := [3][]float64{{0.9793467, 0.69710433, .999572}, {0.9977869}, {0.9905856, 0.9552808}}
	for i, cut := range res.Segments[0].Cuts {
		s_result, err_1 := json.Marshal(cut.Result)
		if err_1 != nil {
			panic(err_1)
		}
		var result TerrorComplexResult
		err_2 := json.Unmarshal(s_result, &result)
		if err_2 == nil {
			Expect(result.Score).Should(BeNumerically("~", ex_score[0][i], 0.001))
			Expect(result.Label).Should(Equal(ex_label[0][i]))
		} else {
			panic(err_2)
		}
	}
}
