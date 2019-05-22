package test

import (
	"encoding/json"
	"fmt"
	"time"

	O "github.com/onsi/ginkgo"
	K "github.com/onsi/gomega"
	P "qiniu.com/argus/test/biz/batch"
	C "qiniu.com/argus/test/biz/client"
	E "qiniu.com/argus/test/biz/env"
	"qiniu.com/argus/test/biz/proto"
	biz "qiniu.com/argus/test/biz/util"
	"qiniu.com/argus/test/lib/qnhttp"
)

func CheckVideoAbnormal(c C.Client, OP string, Path string, groupId string) {
	var res proto.CodeErrorResp
	var req *proto.ArgusVideoRequest
	var resp *qnhttp.Response
	var err error
	var videoName = "test5miao.mp4"
	url := E.Env.GetURIVideo(videoName)
	err = biz.StoreUri("argusvideo/"+videoName, url)
	if err != nil {
		panic(err)
	}
	O.JustBeforeEach(func() {
		if groupId != "" {
			req.SetGroup(0, groupId)
		}
		resp, err = c.PostWithJson(Path, req)
		if err != nil {
			panic(err)
		}
	})
	O.Context("url not exist", func() {
		O.BeforeEach(func() {
			req = proto.NewArgusVideoRequest(E.Env.GetURIVideo("pulp_not_exist.mp4"), OP)
		})
		O.It("ex", func() {
			K.Expect(resp.Status()).To(K.Equal(400))
			K.Expect(json.Unmarshal(resp.BodyToByte(), &res)).Should(K.BeNil())
			K.Expect(res.Code).Should(K.Equal(4000203))
			K.Expect(res.Error).Should(K.Equal("fetch uri failed"))
		})
	})
	O.Context("fetch uri failed", func() {
		var videoName = "bad.mp4"
		url := E.Env.GetURIVideo(videoName)
		err = biz.StoreUri("argusvideo/"+videoName, url)
		if err != nil {
			panic(err)
		}
		O.BeforeEach(func() {
			req = proto.NewArgusVideoRequest(E.Env.GetURIVideo(videoName), OP)
		})
		O.It("ex", func() {
			K.Expect(resp.Status()).To(K.Equal(400))
			K.Expect(json.Unmarshal(resp.BodyToByte(), &res)).Should(K.BeNil())
			K.Expect(res.Code).Should(K.Equal(4000203))
			K.Expect(res.Error).Should(K.Equal("fetch uri failed"))
		})
	})
	O.Context("cannot find video", func() {
		var videoName = "audio-mp3.mp3"
		url := E.Env.GetURIVideo(videoName)
		err = biz.StoreUri("argusvideo/"+videoName, url)
		if err != nil {
			panic(err)
		}
		O.BeforeEach(func() {
			req = proto.NewArgusVideoRequest(E.Env.GetURIVideo(videoName), OP)
		})
		O.It("ex", func() {
			K.Expect(resp.Status()).To(K.Equal(415))
			K.Expect(json.Unmarshal(resp.BodyToByte(), &res)).Should(K.BeNil())
			K.Expect(res.Code).Should(K.Equal(4150501))
			K.Expect(res.Error).Should(K.Equal("cannot find video"))
		})
	})
	O.Context("bad op", func() {
		O.BeforeEach(func() {
			op := "s"
			req = proto.NewArgusVideoRequest(url, op)
		})
		O.It("ex", func() {
			K.Expect(resp.Status()).To(K.Equal(400))
			K.Expect(json.Unmarshal(resp.BodyToByte(), &res)).Should(K.BeNil())
			K.Expect(res.Code).Should(K.Equal(4000100))
			K.Expect(res.Error).Should(K.Equal("bad op"))
		})
	})
	O.Context("no op", func() {
		O.BeforeEach(func() {
			op := ""
			req = proto.NewArgusVideoRequest(url, op)
		})
		O.It("ex", func() {
			K.Expect(resp.Status()).To(K.Equal(400))
			K.Expect(json.Unmarshal(resp.BodyToByte(), &res)).Should(K.BeNil())
			K.Expect(res.Code).Should(K.Equal(4000100))
			K.Expect(res.Error).Should(K.Equal("bad op"))
		})
	})
	O.Context("vframe mode bad", func() {
		O.BeforeEach(func() {
			req = proto.NewArgusVideoRequest(url, OP)
			req.SetVframe(3, 2)
		})
		O.It("ex", func() {
			K.Expect(resp.Status()).To(K.Equal(400))
			K.Expect(json.Unmarshal(resp.BodyToByte(), &res)).Should(K.BeNil())
			K.Expect(res.Code).Should(K.Equal(4000100))
			K.Expect(res.Error).Should(K.Equal("invalid mode, allow mode is [0, 1]"))
		})
	})
}

func CheckVideoSetVframe(c C.Client, OP string, Path string) {
	var req *proto.ArgusVideoRequest
	var resp *qnhttp.Response
	var err error
	O.JustBeforeEach(func() {
		resp, err = c.PostWithJson(Path, req)
		if err != nil {
			panic(err)
		}
	})
	O.Context("set vframe 1", func() {
		var videoName = "test5miao.mp4"
		url := E.Env.GetURIVideo(videoName)
		err = biz.StoreUri("argusvideo/"+videoName, url)
		if err != nil {
			panic(err)
		}
		O.BeforeEach(func() {
			req = proto.NewArgusVideoRequest(url, OP)
			req.SetVframe(0, 2)
		})
		O.It("check vframe", func() {
			var res map[string]proto.VideoOpResult
			K.Expect(json.Unmarshal(resp.BodyToByte(), &res)).Should(K.BeNil())
			K.Expect(resp.Status()).To(K.Equal(200))
			CheckRes(res)
			K.Expect(len(res)).Should(K.Equal(1))
			lengthCut := 0
			for _, segment := range res[OP].Segments {
				lengthCut += len(segment.Cuts)
			}
			O.By("验证固定截帧数")
			K.Expect(lengthCut).Should(K.Equal(3))
		})
	})
	O.Context("set vframe 2", func() {
		var videoName = "pulp_all.mp4"
		url := E.Env.GetURIVideo(videoName)
		err = biz.StoreUri("argusvideo/"+videoName, url)
		if err != nil {
			panic(err)
		}
		O.BeforeEach(func() {
			req = proto.NewArgusVideoRequest(url, OP)
			req.SetVframe(1, 0)
		})
		O.It("check vframe", func() {
			var res map[string]proto.VideoOpResult
			K.Expect(json.Unmarshal(resp.BodyToByte(), &res)).Should(K.BeNil())
			K.Expect(resp.Status()).To(K.Equal(200))
			CheckRes(res)
			K.Expect(len(res)).Should(K.Equal(1))
			lengthCut := 0
			for _, segment := range res[OP].Segments {
				lengthCut += len(segment.Cuts)
			}
			O.By("验证关键截帧数")
			K.Expect(lengthCut).Should(K.Equal(12))
		})
	})
	O.Context("set vframe  more video", func() {
		var videoName = "1536741372944216853video"
		url := E.Env.GetURIVideo(videoName)
		err = biz.StoreUri("argusvideo/"+videoName, url)
		if err != nil {
			panic(err)
		}
		O.BeforeEach(func() {
			req = proto.NewArgusVideoRequest(url, OP)
			req.SetVframe(1, 0)
		})
		O.It("check vframe", func() {
			var res map[string]proto.VideoOpResult
			K.Expect(json.Unmarshal(resp.BodyToByte(), &res)).Should(K.BeNil())
			K.Expect(resp.Status()).To(K.Equal(200))
			CheckRes(res)
			K.Expect(len(res[OP].Segments)).Should(K.Equal(1))
			K.Expect(len(res[OP].Segments[0].Cuts)).Should(K.Equal(4))
		})
	})
	O.Context("fix negtive offset", func() {
		var videoName = "negitive_offset.mp4"
		url := E.Env.GetURIVideo(videoName)
		err = biz.StoreUri("argusvideo/"+videoName, url)
		if err != nil {
			panic(err)
		}
		O.BeforeEach(func() {
			req = proto.NewArgusVideoRequest(url, OP)
			fmt.Println("pos1")
		})
		O.It("ATLAB-8003 fix", func() {
			K.Expect(resp.Status()).To(K.Equal(200))
			var res map[string]proto.VideoOpResult
			K.Expect(json.Unmarshal(resp.BodyToByte(), &res)).Should(K.BeNil())
			K.Expect(len(res[OP].Segments)).Should(K.Equal(1))
			K.Expect(len(res[OP].Segments[0].Cuts)).Should(K.Equal(2))
		})
	})
}

func CheckVideoAsync(c C.Client, OP string, JobId string, JobVid string, JobPath string) (Result map[string]proto.VideoOpsResult) {
	// var req *proto.ArgusVideoRequest
	var resp *qnhttp.Response
	var err error
	var job_res proto.ArgusVideoJob
	//Kill jobs
	defer func() {
		O.By("结束布控任务")
		path_kill := JobPath + JobId + "/kill"
		resp, err = c.Get(JobPath + JobId)
		K.Expect(resp.Status()).To(K.Equal(200))
		K.Expect(err).To(K.BeNil())
		resp.ResponseBodyAsRel(&job_res)
		if job_res.Status != "FINISHED" {
			_, err = c.PostWithJson(path_kill, nil)
			K.Expect(err).Should(K.BeNil())
		}
		K.Expect(resp.Status()).To(K.Equal(200))
		K.Expect(err).To(K.BeNil())
	}()
	for t := 15; t > 0; t-- {
		resp, err = c.Get(JobPath + JobId)
		if err != nil {
			panic(err)
		}
		K.Expect(resp.Status()).To(K.Equal(200))
		resp.ResponseBodyAsRel(&job_res)
		K.Expect(job_res.Id).Should(K.Equal(JobId))
		K.Expect(job_res.Vid).Should(K.Equal(JobVid))
		if job_res.Status == "FINISHED" {
			break
		}

		//K.Expect(job_res.Status).Should(K.Equal("DOING|WAITING"))
		if OP != "face_group_search_private" {
			K.Expect(job_res.Status).ShouldNot(K.Equal("FINISHED"))
		}
		time.Sleep(time.Second * 3)
	}
	return job_res.Result
}

func CheckRes(res map[string]proto.VideoOpResult) {
	for op, res_op := range res {
		K.Expect(op).Should(K.BeAssignableToTypeOf("op"))
		K.Expect(len(res_op.Labels)).Should(K.BeNumerically(">=", 1))
		for _, label := range res_op.Labels {
			K.Expect(label.Label).ShouldNot(K.Equal(""))
			K.Expect(label.Score).Should(K.BeNumerically(">", 0.1))
		}
		K.Expect(len(res_op.Segments)).Should(K.BeNumerically(">=", 1))
		for _, segment := range res_op.Segments {
			K.Expect(segment.Offset_begin).Should(K.BeAssignableToTypeOf(0))
			K.Expect(segment.Offset_end).Should(K.BeAssignableToTypeOf(0))
			K.Expect(len(segment.Labels)).Should(K.BeNumerically(">=", 1))
			flag := -1
			for k, label := range res_op.Labels {
				if segment.Labels[0].Label == label.Label {
					flag = k
				}
			}
			K.Expect(flag).Should(K.BeNumerically(">", -1))
			// K.Expect(segment.Labels[0].Label).Should(K.Equal(res_op.Labels[i].Label))
			for _, labels := range segment.Labels {
				K.Expect(labels.Score).Should(K.BeNumerically("<=", res_op.Labels[flag].Score))
			}
			K.Expect(len(segment.Cuts)).Should(K.BeNumerically(">=", 1))
			for _, cut := range segment.Cuts {
				K.Expect(cut.Offset).Should(K.BeAssignableToTypeOf(0))
				K.Expect(cut.Result).ShouldNot(K.BeNil())
			}
		}
	}
}

func CheckDiffTypeVideo(c C.Client, OP string, Path string, groupId string) {
	var req *proto.ArgusVideoRequest
	var resp *qnhttp.Response
	var err error
	videoList, err := P.BatchGetVideo("AllTypeVideo/")
	if err != nil {
		panic(err)
	}
	O.Context("normal video", func() {
		for _, file := range videoList {
			var fileName = file
			fmt.Println("video" + fileName.Imgname)
			if err := biz.StoreUri("argusvideo/"+fileName.Imgname, fileName.Imgurl); err != nil {
				panic(err)
			}
			O.It("测试视频"+fileName.Imgname, func() {
				req = proto.NewArgusVideoRequest(fileName.Imgurl, OP)
				if groupId != "" {
					req.SetGroup(0, groupId)
				}
				resp, err = c.PostWithJson(Path, req)
				K.Expect(err).Should(K.BeNil())
				K.Expect(resp.Status()).To(K.Or(K.Equal(200), K.Equal(400)))
			})
		}
	})
}
