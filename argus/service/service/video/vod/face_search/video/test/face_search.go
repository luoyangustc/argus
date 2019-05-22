package test

import (
	"encoding/json"
	"fmt"
	"strconv"
	"time"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	Test "qiniu.com/argus/service/service/video/vod/video/test"
	"qiniu.com/argus/test/biz/assert"
	E "qiniu.com/argus/test/biz/env"
	"qiniu.com/argus/test/biz/proto"
	B "qiniu.com/argus/test/biz/util"
	"qiniu.com/argus/test/configs"
	"qiniu.com/argus/test/lib/qnhttp"
	"qiniu.com/argus/test/tool/stub/stublib"
	"qiniu.com/argus/test/tool/util"
)

var _ = Describe("[argus]|[video]|[vod]|[face_group_search_private]", func() {
	var client = E.Env.GetClientArgusVideo()
	var faceClient = E.Env.GetClientFaceCpu()
	var server = "face_search"
	var op = B.GetOp(server, "videovod")
	var jobpath = "/v1/jobs/"
	pathvideo := B.GetPath(server, "videovod", "path") + "/shshsh"
	var req *proto.ArgusVideoRequest
	var resp *qnhttp.Response
	var err error
	var labelName = "name_jin_00001"
	var videoName = []string{"testjin5s.mp4", "politician_s.mp4"}
	//group
	id := 21
	var groupId = "video_test_id" + strconv.Itoa(id)
	var url []string
	//face
	var faceId = "200001"
	var img = "test_jin.jpg"
	var faceuri = E.Env.GetURIPrivate(img)
	//下载资源
	err = B.StoreUri(img, faceuri)
	if err != nil {
		panic(err)
	}
	for _, file := range videoName {
		videoUrl := E.Env.GetURIVideo(file)
		err = B.StoreUri("argusvideo/"+file, videoUrl)
		if err != nil {
			panic(err)
		}
		url = append(url, videoUrl)
	}

	AfterEach(func() {
		path_remove := "/v1/face/groups/" + groupId + "/remove"
		_, err := faceClient.PostWithJson(path_remove, nil)
		if err != nil {
			panic(err)
		}
	})
	BeforeEach(func() {
		req = proto.NewArgusVideoRequest(url[0], op)
		req.SetIgnore(true, 0)
		req.SetVframe(0, 3)
		req.SetGroup(0, groupId)
		// 生成人脸库
		groupPath := "/v1/face/groups/" + groupId
		resp, err := faceClient.PostWithJson(groupPath, NewFaceCreate(100000))
		Expect(err).Should(BeNil())
		Expect(resp.Status()).Should(Equal(200))
		//人脸入库
		path2 := "/v1/face/groups/" + groupId + "/add"
		inputData2 := NewFaceAdd(faceId, faceuri, labelName, FaceDesc{"jinzhengen", faceId})
		resp2, err2 := faceClient.PostWithJson(path2, inputData2)
		Expect(resp2.Status()).Should(Equal(200))
		Expect(err2).Should(BeNil())
	})
	Describe("Sync detection", func() {
		Context("Basic", func() {
			It("Normal", func() {
				//检索人脸
				resp, err = client.PostWithJson(pathvideo, req)
				if err != nil {
					panic(err)
				}
				var res map[string]proto.VideoOpResult
				Expect(json.Unmarshal(resp.BodyToByte(), &res)).Should(BeNil())
				Expect(resp.Status()).To(Equal(200))
				Test.CheckRes(res)
				CheckNormalFaceSearchResp(res[op], op, labelName)
			})
		})
	})
	Describe("Abnormal test", func() {
		Test.CheckVideoAbnormal(client, op, pathvideo, groupId)
	})
	Describe("Check Vframe", func() {
		JustBeforeEach(func() {
			req.ChangeVideoUrl(url[1])
			resp, err = client.PostWithJson(pathvideo, req)
			if err != nil {
				panic(err)
			}
		})
		Context("set Vframe 0,3", func() {
			var interval = 3
			BeforeEach(func() {
				req.SetVframe(0, interval)
			})
			It("Normal all", func() {
				var res map[string]proto.VideoOpResult
				Expect(resp.Status()).To(Equal(200))
				Expect(json.Unmarshal(resp.BodyToByte(), &res)).To(BeNil())
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
		Context("set Vframe 1,0", func() {
			var interval = 5
			BeforeEach(func() {
				req.SetVframe(1, 0)
				//默认截帧 5.0s
			})
			It("Normal all", func() {
				var res map[string]proto.VideoOpResult
				Expect(resp.Status()).To(Equal(200))
				Expect(json.Unmarshal(resp.BodyToByte(), &res)).To(BeNil())
				Expect(len(res[op].Segments)).Should(BeNumerically(">=", 0))
				Expect(res[op].Segments[0].Offset_begin).Should(BeNumerically(">", 0))
				Expect(res[op].Segments[0].Offset_end).Should(BeNumerically(">=", res[op].Segments[0].Offset_begin))
				var firstCut = res[op].Segments[0].Cuts[0].Offset
				var id = 0
				for _, segment := range res[op].Segments {
					for _, cut := range segment.Cuts {
						if id < 2 {
							Expect(cut.Offset).Should(Equal(firstCut + id*interval*1000))
						}
						id++
					}
				}
			})
		})
	})
	Describe("Async Face_search Job", func() {
		if !configs.StubConfigs.Servers.VideoAsync {
			return
		}
		var job_id string
		Context("normal", func() {
			var job_vid = "face_search_job"
			JustBeforeEach(func() {
				pathvideo = B.GetPath(server, "videovod", "path") + "/" + job_vid + "/async"
				req = proto.NewArgusVideoRequest(url[0], op)
				req.SetGroup(0, groupId)
				req.SetVframe(0, 3)
				// req.SetLive(5.0, url[0])
				resp, err = client.PostWithJson(pathvideo, req)
				if err != nil {
					panic(err)
				}
			})
			It("Basic", func() {
				Expect(resp.Status()).To(Equal(200))
				job_id = resp.ResponseBodyAsJson()["job"].(string)
				Res := Test.CheckVideoAsync(client, op, job_id, job_vid, jobpath)
				CheckNormalFaceSearchResp(Res[op].Result, op, labelName)
			})
		})
		Context("Get Job", func() {
			var job_vid = "face_search_video_jobs"
			var job_ids []string
			var BeginJobsLen int
			JustBeforeEach(func() {
				var job_ress Job_Res
				resp, err = client.Get(string([]byte(jobpath)[:len(jobpath)-1]))
				Expect(resp.Status()).To(Equal(200))
				Expect(err).To(BeNil())
				Expect(json.Unmarshal(resp.BodyToByte(), &job_ress)).Should(BeNil())
				BeginJobsLen = len(job_ress.Jobs)
				for i := 0; i < 3; i++ {
					pathLive := B.GetPath(server, "videovod", "path") + "/" + job_vid + strconv.Itoa(i) + "/async"
					req := proto.NewArgusLiveRequest(url[0], op)
					req.SetLive(5)
					req.SetLiveVframe(0, 3)
					req.SetLimitWithGroup(3, []string{groupId}, 0)
					// req.SetLive(5.0, url[0])
					resp, err = client.PostWithJson(pathLive, req)
					if err != nil {
						panic(err)
					}
					job_id := resp.ResponseBodyAsJson()["job"].(string)
					job_ids = append(job_ids, job_id)
				}
			})
			It("查询异步任务", func() {
				defer func() {
					var job_res proto.ArgusVideoJob
					for _, job := range job_ids {
						path_kill := jobpath + job + "/kill"
						resp, err = client.Get(jobpath + job)
						resp.ResponseBodyAsRel(&job_res)
						if job_res.Status != "FINISHED" {
							_, err = client.PostWithJson(path_kill, nil)
							Expect(err).Should(BeNil())
						}
						Expect(resp.Status()).To(Equal(200))
					}
				}()
				Jobpath := string([]byte(jobpath)[:len(jobpath)-1])
				By("查询全部")
				var job_ress1 Job_Res
				resp, err = client.Get(Jobpath)
				Expect(resp.Status()).To(Equal(200))
				Expect(err).To(BeNil())
				Expect(json.Unmarshal(resp.BodyToByte(), &job_ress1)).Should(BeNil())
				Expect(len(job_ress1.Jobs)).Should(Equal(len(job_ids) + BeginJobsLen))
				By("返回数目限制")
				var job_ress2 Job_Res
				resp, err = client.Get(Jobpath + "?limit=2")
				Expect(resp.Status()).To(Equal(200))
				Expect(err).To(BeNil())
				Expect(json.Unmarshal(resp.BodyToByte(), &job_ress2)).Should(BeNil())
				Expect(len(job_ress2.Jobs)).Should(Equal(2))
				var marker = job_ress2.Marker
				By("Marker验证")
				var job_ress3 Job_Res
				resp, err = client.Get(Jobpath + "?limit=1&marker=" + marker)
				Expect(resp.Status()).To(Equal(200))
				Expect(err).To(BeNil())
				Expect(json.Unmarshal(resp.BodyToByte(), &job_ress3)).Should(BeNil())
				Expect(len(job_ress3.Jobs)).Should(Equal(1))
				By("Status验证")
				var job_ress4 Job_Res
				resp, err = client.Get(Jobpath + "?status=DOING")
				Expect(resp.Status()).To(Equal(200))
				Expect(err).To(BeNil())
				Expect(json.Unmarshal(resp.BodyToByte(), &job_ress4)).Should(BeNil())
				for _, Job := range job_ress4.Jobs {
					Expect(Job.Status).Should(Equal("DOING"))
				}
			})
		})
		Context("Call back", func() {
			var stub *stublib.LiveStub
			var job_vid = "face_sreach_callback"
			var limit = 3
			JustBeforeEach(func() {
				var domain string
				port, err := util.AllocPort()
				if err != nil {
					panic(err)
				}
				host, _ := util.Localhost()
				if host != "" {
					domain = "http://" + host + ":" + port
				}
				By("stub" + domain)
				//
				stub = stublib.NewLiveStub(port)
				stub.SetRes(`{"errno":0}`)

				pathLive := B.GetPath(server, "videovod", "path") + "/" + job_vid + "/async"
				req := proto.NewArgusLiveRequest(url[0], op)
				req.SetLive(5)
				req.SetLiveVframe(0, 3)
				req.SetLiveHookUrl(domain, 0)
				req.SetLimitWithGroup(limit, []string{groupId}, 0)
				// req.SetLive(5.0, url[0])
				resp, err = client.PostWithJson(pathLive, req)
				if err != nil {
					panic(err)
				}
			})
			It("Basic", func() {
				defer func() {
					//server 随程序结束 close
					fmt.Println("---------------------------------Stub--Close-------------------------------")
				}()
				Expect(resp.Status()).To(Equal(200))
				job_id = resp.ResponseBodyAsJson()["job"].(string)
				//视频循环无FININSHED
				_ = Test.CheckVideoAsync(client, op, job_id, job_vid, jobpath)
				By("推理结果")
				var errtime int = 0
				for i := 1; i < 10; i++ {
					var job_res LiveOpsRes
					if stub.Request == nil {
						time.Sleep(time.Second * 3)
						fmt.Printf("%d", i)
					} else {
						reslen := len(stub.Livemock.Request)
						if reslen > 100 {
							reslen = 100
						}
						for i := 0; i < reslen; i++ {
							Expect(json.Unmarshal(stub.Livemock.Request[i], &job_res)).To(BeNil())
							if job_res.Job_id != job_id && errtime < 5 {
								errtime++
							} else {
								//验证推理部署与回调一致
								Expect(errtime < 5).To(Equal(true))
								Expect(job_res.Job_id).Should(Equal(job_id))
								Expect(job_res.Live_id).Should(Equal(job_vid))
								Expect(job_res.Op).Should(Equal(op))
								//验证face回调limit
								var LenFaces = 0
								for a := 0; a < len(job_res.Result.Faces); a++ {
									LenFaces += len(job_res.Result.Faces[a].Faces)
								}
								Expect(LenFaces).Should(BeNumerically("<=", limit))
							}
						}

					}
				}
			})

		})
	})
	Describe("Normal", func() {
		Test.CheckDiffTypeVideo(client, op, pathvideo, groupId)
	})
})

//
type Job_Res struct {
	Jobs   []proto.ArgusVideoJob `json:"jobs"`
	Marker string                `json:"marker,omitempty"`
}

//Callback Resp
type FacesBounding_Box struct {
	Pts   [][2]int `json:"pts"`
	Score float64  `json:"score"`
}

type Faces_faces struct {
	Id    string      `json:"id"`
	Score float64     `json:"score"`
	Tag   string      `json:"tag"`
	Desc  interface{} `json:"desc"`
}

type ResultFaces struct {
	Bounding_box FacesBounding_Box `json:"bounding_box"`
	Faces        []Faces_faces     `json:"faces"`
}

type OpResult struct {
	Faces []ResultFaces `json:"faces"`
}

type LiveOpsRes struct {
	Id      string   `json:"id,omitempty"`
	Job_id  string   `json:"job_id"`
	Live_id string   `json:"live_id,omitempty"`
	Op      string   `json:"op"`
	Offset  int      `json:"offset"`
	Uri     string   `json:"uri,omitempty"`
	Result  OpResult `json:"result"`
}

//Facegroup Search
type FaceGroupSearchRes struct {
	Detections []struct {
		BoundingBox struct {
			Pts   [][2]int `json:"pts"`
			Score float64  `json:"score"`
		} `json:"boundingBox"`
		Vlaue struct {
			Score  float64 `json:"score"`
			Name   string  `json:"name"`
			Review bool    `json:"review"`
		} `json:"value"`
	} `json:"detections"`
}

//Facegroup Create
type FaceCreate struct {
	Config FaceConfig `json:"config"`
}

type FaceConfig struct {
	Capacity  int `json:"capacity,omitempty"`
	Dimension int `json:"dimension,omitempty"`
	Precision int `json:"precision,omitempty"`
	Version   int `json:"version,omitempty"`
}

func NewFaceCreate(capacity int) *FaceCreate {
	return &FaceCreate{
		Config: FaceConfig{
			Capacity: capacity,
		},
	}
}

//Add Face
type FaceAdd struct {
	Image  FaceImage       `json:"image"`
	Params FaceImageParams `json:"Params,omitempty"`
}
type FaceDesc struct {
	Name string `json:"name"`
	Id   string `json:"id"`
}

type FaceImage struct {
	Id   string      `json:"id"`
	Uri  string      `json:"uri"`
	Tag  string      `json:"tag"`
	Desc interface{} `json:"desc,omitempty"`
	//Bounding_box BoundingBox `json:"bounding_box,omitempty"`
}

type FaceImageSRes struct {
	Id           string      `json:"id"`
	Uri          string      `json:"uri"`
	Tag          string      `json:"tag"`
	Desc         FaceDesc    `json:"desc,omitempty"`
	Pounding_box BoundingBox `json:"bounding_box,omitempty"`
}

type BoundingBox struct {
	Pts [4][2]int `json:"pts,omitempty"`
}

type FaceImageParams struct {
	Reject_bad_face bool `json:"reject_bad_face"`
}

func NewFaceAdd(id string, uri string, tag string, desc interface{}) *FaceAdd {
	return &FaceAdd{
		Image: FaceImage{
			Id:   id,
			Uri:  uri,
			Tag:  tag,
			Desc: desc,
		},
		Params: FaceImageParams{Reject_bad_face: false},
	}
}

func CheckNormalFaceSearchResp(res proto.VideoOpResult, op string, label string) {
	Expect(len(res.Labels)).Should(Equal(1))
	Expect(len(res.Segments)).Should(Equal(1))
	Expect(res.Labels[0].Score).Should(BeNumerically(">", 0.8))
	Expect(res.Labels[0].Label).Should(Equal(label))
	Expect(res.Segments[0].Offset_begin).Should(Equal(33))
	Expect(res.Segments[0].Offset_end).Should(Equal(3002))
	Expect(len(res.Segments[0].Cuts)).Should(Equal(2))
	for _, cut := range res.Segments[0].Cuts {
		s_result, err_1 := json.Marshal(cut.Result)
		if err_1 != nil {
			panic(err_1)
		}
		var results FaceGroupSearchRes
		err_2 := json.Unmarshal(s_result, &results)
		if err_2 == nil {
			for _, result := range results.Detections {
				fmt.Println(result.BoundingBox.Pts)
				By("Pts验证")
				assert.CheckPts(result.BoundingBox.Pts)
				Expect(result.BoundingBox.Score).Should(BeNumerically(">", 0.99))
				Expect(result.Vlaue.Name).Should(Equal(label))
				Expect(result.Vlaue.Score).Should(BeNumerically(">", 0.8))
				Expect(result.Vlaue.Review).Should(Equal(false))
			}
		} else {
			panic(err_2)
		}
	}
}
