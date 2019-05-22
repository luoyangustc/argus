package evaltest

import (
	"encoding/json"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	E "qiniu.com/argus/test/biz/env"
	"qiniu.com/argus/test/biz/proto"
	T "qiniu.com/argus/test/biz/tsv"
	biz "qiniu.com/argus/test/biz/util"
	"qiniu.com/argus/test/configs"
)

func EvalAdsClassifier(EvalServer string, Server string, SceneType string) {
	params, err := proto.NewAttr(`{"type":["ads","pulp","terror","politician","other"]}`)
	if err != nil {
		panic(err)
	}
	var (
		client = E.Env.GetClientServing()
		path   = biz.GetPath(Server, SceneType, "evalpath")
		tsv    = T.NewTsv(configs.StubConfigs.Servers.Type["eval"][EvalServer].Tsv,
			configs.StubConfigs.Servers.Type["eval"][EvalServer].Set, path, configs.StubConfigs.Servers.Type["eval"][EvalServer].Precision,
			params)
	)
	Describe("ads-classifier tsv", func() {
		Context("ads-classifier tsv", func() {
			TsvClassifierTest(client, tsv, T.CheckCommon)
		})
	})
	Describe("special text", func() {
		It("text is null", func() {
			texts := []string{}
			uriBase64, err := TextToBase64(texts)
			if err != nil {
				panic(err)
			}
			resp, err := client.PostWithJson(path,
				proto.NewArgusReq(uriBase64, nil, params))
			Expect(err).Should(BeNil())
			var resObj AdsClassifierRes
			Expect(json.Unmarshal(resp.BodyToByte(), &resObj))
			Expect(len(resObj.Result)).To(Equal(5))
		})
		It("summary逻辑验证", func() {
			texts := []string{"上门服务", "唯品会"}
			uriBase64, err := TextToBase64(texts)
			if err != nil {
				panic(err)
			}
			resp, err := client.PostWithJson(path,
				proto.NewArgusReq(uriBase64, nil, params))
			Expect(err).Should(BeNil())
			var resObj AdsClassifierRes
			Expect(json.Unmarshal(resp.BodyToByte(), &resObj))
			Expect(len(resObj.Result)).To(Equal(5))
			for _, tpRes := range resObj.Result {
				exSummary := GetSummary(tpRes.Confidences)
				Expect(tpRes.Summary.Score).To(BeNumerically("~", 1, 0.000001))
				Expect(tpRes.Summary.Label).To(Equal(exSummary.Label))
			}
		})
		It("重复text", func() {
			texts := []string{"上门服务", "唯品会", "上门服务", "唯品会"}
			uriBase64, err := TextToBase64(texts)
			if err != nil {
				panic(err)
			}
			resp, err := client.PostWithJson(path,
				proto.NewArgusReq(uriBase64, nil, params))
			Expect(err).Should(BeNil())
			var resObj AdsClassifierRes
			Expect(json.Unmarshal(resp.BodyToByte(), &resObj))
			Expect(len(resObj.Result)).To(Equal(5))
			for _, tpRes := range resObj.Result {
				exSummary := GetSummary(tpRes.Confidences)
				// Expect(tpRes.Summary.Score).To(BeNumerically("~", exSummary.Score, 0.000001))
				Expect(tpRes.Summary.Label).To(Equal(exSummary.Label))
			}
			Expect(len(resObj.Result["ads"].Confidences)).To(Equal(4))
		})
		It("空text结果不被忽略", func() {
			texts := []string{"上门服务", "", "上门服务", "唯品会"}
			uriBase64, err := TextToBase64(texts)
			if err != nil {
				panic(err)
			}
			resp, err := client.PostWithJson(path,
				proto.NewArgusReq(uriBase64, nil, params))
			Expect(err).Should(BeNil())
			var resObj AdsClassifierRes
			Expect(json.Unmarshal(resp.BodyToByte(), &resObj))
			Expect(len(resObj.Result)).To(Equal(5))
			for _, tpRes := range resObj.Result {
				exSummary := GetSummary(tpRes.Confidences)
				// Expect(tpRes.Summary.Score).To(BeNumerically("~", exSummary.Score, 0.000001))
				Expect(tpRes.Summary.Label).To(Equal(exSummary.Label))
			}
			Expect(len(resObj.Result["ads"].Confidences)).To(Equal(4))
		})
		It("重复keys", func() {
			texts := []string{"上门服务唯品会上门服务唯品会", "唯品会", "上门服务", "唯品会"}
			uriBase64, err := TextToBase64(texts)
			if err != nil {
				panic(err)
			}
			resp, err := client.PostWithJson(path,
				proto.NewArgusReq(uriBase64, nil, params))
			Expect(err).Should(BeNil())
			var resObj AdsClassifierRes
			Expect(json.Unmarshal(resp.BodyToByte(), &resObj))
			Expect(len(resObj.Result)).To(Equal(5))
			for _, tpRes := range resObj.Result {
				exSummary := GetSummary(tpRes.Confidences)
				// Expect(tpRes.Summary.Score).To(BeNumerically("~", exSummary.Score, 0.000001))
				Expect(tpRes.Summary.Label).To(Equal(exSummary.Label))
			}
			Expect(len(resObj.Result["ads"].Confidences)).To(Equal(4))
		})
		It("text contain *?&%/'\\", func() {
			texts := []string{"上门服务'*?&/\\"}
			uriBase64, err := TextToBase64(texts)
			if err != nil {
				panic(err)
			}
			resp, err := client.PostWithJson(path,
				proto.NewArgusReq(uriBase64, nil, params))
			Expect(err).Should(BeNil())
			var resObj AdsClassifierRes
			Expect(json.Unmarshal(resp.BodyToByte(), &resObj))
			Expect(len(resObj.Result)).To(Equal(5))
		})
		It("大写字母fix", func() {
			texts := []string{"代充Q币"}
			uriBase64, err := TextToBase64(texts)
			if err != nil {
				panic(err)
			}
			resp, err := client.PostWithJson(path,
				proto.NewArgusReq(uriBase64, nil, params))
			Expect(err).Should(BeNil())
			var resObj AdsClassifierRes
			Expect(json.Unmarshal(resp.BodyToByte(), &resObj))
			Expect(len(resObj.Result)).To(Equal(5))
			Expect(resObj.Result["ads"].Summary.Label).To(Equal("ads"))
		})
		It("编码问题fix ATLAB-10298", func() {
			texts := []string{"73784366", "办证刻章", "办证刻章", "”意后付款", "-6的"}
			uriBase64, err := TextToBase64(texts)
			if err != nil {
				panic(err)
			}
			resp, err := client.PostWithJson(path,
				proto.NewArgusReq(uriBase64, nil, params))
			Expect(err).Should(BeNil())
			var resObj AdsClassifierRes
			Expect(json.Unmarshal(resp.BodyToByte(), &resObj))
			Expect(len(resObj.Result)).To(Equal(5))
			Expect(resObj.Result["ads"].Summary.Label).To(Equal("ads"))
		})
	})
}
