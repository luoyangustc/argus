package assert

import (
	"encoding/json"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	C "qiniu.com/argus/test/biz/client"
	E "qiniu.com/argus/test/biz/env"
	"qiniu.com/argus/test/biz/proto"
	biz "qiniu.com/argus/test/biz/util"
	"qiniu.com/argus/test/lib/qnhttp"
)

func CheckImageCode(c C.Client, Path string, Params interface{}) {
	var (
		resp      *qnhttp.Response
		err       error
		imgset    = "test/image/normal/"
		fileNames = []string{"wrongType-201902131804.txt", "big-201807121447.jpg", "big-201810181124.jpeg", "difftype-201807121454.png"}
		urls      []string
	)
	for _, img := range fileNames {
		fileName := img
		uri := E.Env.GetURIPrivate(imgset + fileName)
		if err := biz.StoreUri(imgset+fileName, uri); err != nil {
			panic(err)
		}
		urls = append(urls, uri)
	}
	It("code 4000201", func() {
		var res CodeResp
		var url = "rtmp://not.supported.url"
		resp, err = c.PostWithJson(Path,
			proto.NewArgusReq(url, nil, Params))
		Expect(err).Should(BeNil())
		Expect(resp.Status()).To(Equal(400))
		Expect(json.Unmarshal(resp.BodyToByte(), &res)).Should(BeNil())
		Expect(res.Error).Should(Equal("uri not supported: rtmp://not.supported.url"))
		Expect(res.Code).Should(Equal(4000201))
	})
	It("code 4000203", func() {
		var url = "http://fetch.uri.failed"
		var res CodeResp
		resp, err = c.PostWithJson(Path,
			proto.NewArgusReq(url, nil, Params))
		Expect(err).Should(BeNil())
		Expect(resp.Status()).To(Equal(400))
		Expect(json.Unmarshal(resp.BodyToByte(), &res)).Should(BeNil())
		Expect(res.Error).Should(Equal("fetch uri failed: http://fetch.uri.failed"))
		Expect(res.Code).Should(Equal(4000203))
	})
	It("code 4150301", func() {
		var res CodeResp
		resp, err = c.PostWithJson(Path,
			proto.NewArgusReq(urls[0], nil, Params))
		Expect(err).Should(BeNil())
		Expect(resp.Status()).To(Equal(415))
		Expect(json.Unmarshal(resp.BodyToByte(), &res)).Should(BeNil())
		Expect(res.Error).Should(Equal("not image"))
		Expect(res.Code).Should(Equal(4150301))
	})
	It("code 4000302 :test1", func() {
		var res CodeResp
		resp, err = c.PostWithJson(Path,
			proto.NewArgusReq(urls[1], nil, Params))
		Expect(err).Should(BeNil())
		Expect(resp.Status()).To(Equal(400))
		Expect(json.Unmarshal(resp.BodyToByte(), &res)).Should(BeNil())
		Expect(res.Error).Should(Equal("image is too large, should be in 4999x4999"))
		Expect(res.Code).Should(Equal(4000302))
	})
	It("code 4000302 :test2", func() {
		var res CodeResp
		resp, err = c.PostWithJson(Path,
			proto.NewArgusReq(urls[2], nil, Params))
		Expect(err).Should(BeNil())
		Expect(resp.Status()).To(Equal(400))
		Expect(json.Unmarshal(resp.BodyToByte(), &res)).Should(BeNil())
		Expect(res.Error).Should(Equal("image is too large, should be less than 10MB"))
		Expect(res.Code).Should(Equal(4000302))
	})
	It("code 4000100 :test1", func() {
		var res CodeResp
		if Path != biz.GetPath("censorimage", "image", "path") {
			return
		} else {
			resp, err = c.PostWithJson(Path,
				proto.NewArgusReq(urls[3], nil, nil))
			Expect(err).Should(BeNil())
			Expect(resp.Status()).To(Equal(400))
			Expect(json.Unmarshal(resp.BodyToByte(), &res)).Should(BeNil())
			Expect(res.Error).Should(Equal("empty scene"))
			Expect(res.Code).Should(Equal(4000100))
		}
	})
	It("code 4000100 :test2", func() {
		var res CodeResp
		if Path != biz.GetPath("censorimage", "image", "path") {
			return
		} else {
			Params := map[string][]string{"scenes": []string{"bad"}}
			resp, err = c.PostWithJson(Path,
				proto.NewArgusReq(urls[3], nil, Params))
			Expect(err).Should(BeNil())
			Expect(resp.Status()).To(Equal(400))
			Expect(json.Unmarshal(resp.BodyToByte(), &res)).Should(BeNil())
			Expect(res.Error).Should(Equal("bad scene"))
			Expect(res.Code).Should(Equal(4000100))
		}
	})
}

type CodeResp struct {
	Code  int    `json:"code"`
	Error string `json:"error"`
}
