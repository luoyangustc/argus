package test

import (
	"encoding/json"
	"fmt"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	censor "qiniu.com/argus/service/service/image/censor"
	"qiniu.com/argus/test/biz/assert"
	E "qiniu.com/argus/test/biz/env"
	"qiniu.com/argus/test/biz/proto"
	biz "qiniu.com/argus/test/biz/util"
	"qiniu.com/argus/test/configs"
)

var _ = Describe("[argus]|[image]/v1/image/censor", func() {
	var (
		client         = E.Env.GetClientArgus()
		server         = "imagecensor"
		path           = biz.GetPath(server, "image", "path")
		typePulp       = configs.StubConfigs.Servers.Online.ImagePulp
		typeTerror     = configs.StubConfigs.Servers.Online.ImageTerror
		typePolitician = configs.StubConfigs.Servers.Online.ImagePolitician
		censorType     []string
	)
	if typePolitician {
		fmt.Println("chose politician")
		censorType = append(censorType, "politician")
	}
	if typeTerror {
		fmt.Println("chose terror")
		censorType = append(censorType, "terror")
	}
	if typePulp {
		fmt.Println("chose pulp")
		censorType = append(censorType, "pulp")
	}
	Describe(fmt.Sprintf("%s", censorType), func() {
		Context("gif图片", func() {
			fileName := "upload.gif"
			imguri := E.Env.GetURIPrivate(fileName)
			err := biz.StoreUri(fileName, imguri)
			if err != nil {
				panic(err)
			}
			It("测试gif图片"+fileName, func() {
				url := imguri

				param := map[string][]string{"type": censorType}
				resp, err := client.PostWithJson(path,
					proto.NewArgusReq(url, nil, param))
				Expect(err).Should(BeNil())
				Expect(resp.Status()).To(Equal(200))
				var actual censor.ImageCensorResp
				err = json.Unmarshal(resp.BodyToByte(), &actual)
				Expect(err).Should(BeNil())
				var censortypecount = 0
				for _, detail := range actual.Result.Details {
					for _, centype := range censorType {
						if detail.Type == centype {
							censortypecount++
						}
					}
				}
				Expect(censortypecount).To(Equal(len(actual.Result.Details)))
			})
		})
	})
	Describe("pulp", func() {
		if !typePulp {
			return
		}
		Context("normal图片", func() {
			fileName := "pulpnormal.jpg"
			imguri := E.Env.GetURIPrivate(fileName)
			err := biz.StoreUri(fileName, imguri)
			if err != nil {
				panic(err)
			}
			It("测试图片"+fileName, func() {
				url := imguri
				param := map[string][]string{"type": {"pulp"}}
				resp, err := client.PostWithJson(path,
					proto.NewArgusReq(url, nil, param))
				Expect(err).Should(BeNil())
				Expect(resp.Status()).To(Equal(200))
				var actual proto.ArgusResponse
				err = json.Unmarshal(resp.BodyToByte(), &actual)
				Expect(err).Should(BeNil())
				actA, ok := actual.Result.(map[string]interface{})
				Expect(ok).Should(Equal(true))
				Expect(actA["label"]).Should(Equal(0.0))

			})
		})
		Context("pulp图片", func() {
			fileName := "pulppulp.jpg"
			imguri := E.Env.GetURIPrivate(fileName)
			err := biz.StoreUri(fileName, imguri)
			if err != nil {
				panic(err)
			}
			It("测试图片"+fileName, func() {
				uri := imguri
				param := SetCensorParams([]string{"pulp"}, false)
				resp, err := client.PostWithJson(path,
					proto.NewArgusReq(uri, nil, param))
				Expect(err).Should(BeNil())
				Expect(resp.Status()).To(Equal(200))
				var actual proto.ArgusResponse
				err = json.Unmarshal(resp.BodyToByte(), &actual)
				Expect(err).Should(BeNil())
				actA, ok := actual.Result.(map[string]interface{})
				Expect(ok).Should(Equal(true))
				Expect(actA["label"]).Should(Equal(1.0))
			})
		})
		Context("sexy图片", func() {
			fileName := "pulpsexy.jpg"
			imguri := E.Env.GetURIPrivate(fileName)
			err := biz.StoreUri(fileName, imguri)
			if err != nil {
				panic(err)
			}
			It("测试图片"+fileName, func() {
				uri := imguri
				param := SetCensorParams([]string{"pulp"}, true)
				resp, err := client.PostWithJson(path,
					proto.NewArgusReq(uri, nil, param))
				Expect(err).Should(BeNil())
				Expect(resp.Status()).To(Equal(200))
				var actual proto.ArgusResponse
				err = json.Unmarshal(resp.BodyToByte(), &actual)
				Expect(err).Should(BeNil())
				actA, ok := actual.Result.(map[string]interface{})
				Expect(ok).Should(Equal(true))
				Expect(actA["label"]).Should(Equal(0.0))
			})
			It("detail string test"+fileName, func() {
				uri := imguri
				param := SetParamsDetailString([]string{"pulp"})
				resp, err := client.PostWithJson(path,
					proto.NewArgusReq(uri, nil, param))
				Expect(err).Should(BeNil())
				Expect(resp.Status()).To(Equal(599))
				msg := resp.ResponseBodyAsJson()["error"].(string)
				Expect(msg).Should(Equal("json: cannot unmarshal string into Go struct field .detail of type bool"))
			})
		})

	})
	Describe("politician", func() {
		if !typePolitician {
			return
		}
		Context("非政治人物", func() {
			fileName := "luhan.jpeg"
			imguri := E.Env.GetURIPrivate(fileName)
			err := biz.StoreUri(fileName, imguri)
			if err != nil {
				panic(err)
			}
			It("测试图片"+fileName, func() {
				uri := imguri
				param := SetCensorParams([]string{"politician"}, false)
				resp, err := client.PostWithJson(path,
					proto.NewArgusReq(uri, nil, param))
				Expect(err).Should(BeNil())
				Expect(resp.Status()).To(Equal(200))
				var resObj proto.ArgusRes
				Expect(json.Unmarshal(resp.BodyToByte(), &resObj)).Should(BeNil())
				Expect(resObj.Result.Label).Should(Equal(0))
				Expect(resObj.Result.Score).Should(BeNumerically(">", 0.6))
			})
		})
		Context("政治人物", func() {
			fileName := "test/image/politician/maozhuxi_yangshangkun.jpg"
			imguri := E.Env.GetURIPrivate(fileName)
			err := biz.StoreUri(fileName, imguri)
			if err != nil {
				panic(err)
			}
			It("测试图片"+fileName, func() {
				uri := imguri
				param := SetCensorParams([]string{"politician"}, true)
				resp, err := client.PostWithJson(path,
					proto.NewArgusReq(uri, nil, param))
				Expect(err).Should(BeNil())
				Expect(resp.Status()).To(Equal(200))
				var resObj proto.ArgusRes
				Expect(json.Unmarshal(resp.BodyToByte(), &resObj)).Should(BeNil())
				Expect(resObj.Result.Label).Should(Equal(1))
				Expect(resObj.Result.Score).Should(BeNumerically(">", 0.6))
			})
			It("detail string test"+fileName, func() {
				uri := imguri
				param := SetParamsDetailString([]string{"politician"})
				resp, err := client.PostWithJson(path,
					proto.NewArgusReq(uri, nil, param))
				Expect(err).Should(BeNil())
				Expect(resp.Status()).To(Equal(599))
				msg := resp.ResponseBodyAsJson()["error"].(string)
				Expect(msg).Should(Equal("json: cannot unmarshal string into Go struct field .detail of type bool"))
			})
		})

	})
	Describe("terror", func() {
		if !typeTerror {
			return
		}
		Context("normal图片", func() {
			fileName := "normaltest.jpeg"
			imguri := E.Env.GetURIPrivate(fileName)
			err := biz.StoreUri(fileName, imguri)
			if err != nil {
				panic(err)
			}
			It("测试图片"+fileName, func() {
				uri := imguri
				param := SetCensorParams([]string{"terror"}, false)
				resp, err := client.PostWithJson(path,
					proto.NewArgusReq(uri, nil, param))
				Expect(err).Should(BeNil())
				Expect(resp.Status()).To(Equal(200))
				var actual proto.ArgusResponse
				err = json.Unmarshal(resp.BodyToByte(), &actual)
				Expect(err).Should(BeNil())
				actA, ok := actual.Result.(map[string]interface{})
				Expect(ok).Should(Equal(true))
				Expect(actA["label"]).Should(Equal(0.0))
			})
		})
		Context("terror图片", func() {
			fileName := "bloodiness.jpeg"
			imguri := E.Env.GetURIPrivate(fileName)
			err := biz.StoreUri(fileName, imguri)
			if err != nil {
				panic(err)
			}
			It("测试图片"+fileName, func() {
				uri := imguri
				param := SetCensorParams([]string{"terror"}, true)
				resp, err := client.PostWithJson(path,
					proto.NewArgusReq(uri, nil, param))
				Expect(err).Should(BeNil())
				Expect(resp.Status()).To(Equal(200))
				var actual proto.ArgusResponse
				err = json.Unmarshal(resp.BodyToByte(), &actual)
				Expect(err).Should(BeNil())
				actA, ok := actual.Result.(map[string]interface{})
				Expect(ok).Should(Equal(true))
				Expect(actA["label"]).Should(Equal(1.0))
			})
			It("detail string test"+fileName, func() {
				uri := imguri
				param := SetParamsDetailString([]string{"terror"})
				resp, err := client.PostWithJson(path,
					proto.NewArgusReq(uri, nil, param))
				Expect(err).Should(BeNil())
				Expect(resp.Status()).To(Equal(599))
				msg := resp.ResponseBodyAsJson()["error"].(string)
				Expect(msg).Should(Equal("json: cannot unmarshal string into Go struct field .detail of type bool"))
			})
		})

	})
	Describe("反向用例", func() {
		Context("错误码校验", func() {
			params := map[string][]string{"type": censorType}
			assert.CheckImageCode(client, path, params)
		})
	})
})

//v1/censor/image  Params
type Params struct {
	Type   []string `json:"type"`
	Detail bool     `json:"detail"`
}

func SetCensorParams(paramsType []string, detail bool) Params {
	var params Params
	params.Type = paramsType
	params.Detail = detail
	return params
}

func SetParamsDetailString(paramsType []string) interface{} {
	var params struct {
		Type   []string `json:"type"`
		Detail string   `json:"detail"`
	}
	params.Type = paramsType
	params.Detail = "true"
	return params
}
