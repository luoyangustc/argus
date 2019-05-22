package test

import (
	"fmt"

	// "io/ioutil"

	"encoding/json"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	assert "qiniu.com/argus/test/biz/assert"
	"qiniu.com/argus/test/biz/env"
	E "qiniu.com/argus/test/biz/env"
	"qiniu.com/argus/test/biz/proto"
	biz "qiniu.com/argus/test/biz/util"
	"qiniu.com/argus/test/configs"
)

var _ = Describe("锐安银川网安定制图像类型分类模型|/v1/ocr/classify", func() {
	var server = "ocryinchuan"
	var path = biz.GetPath(server, "image", "path")
	var client = E.Env.GetClientArgus()

	Describe("正向用例", func() {
		imgs := []string{
			"0_bank_card6.jpg",
			"4_cachet6.jpg",
			"13.jpg",
			"bankidcard.jpeg",
			"46_wxy.jpg",
			"49_normal_screen_mobile_1101_07616.jpg",
			"9764bb42gy1flhmf2lqavj20ku45uasl.jpg",
			"WechatIMG66.png",
			"WechatIMG279.jpeg",
		}
		precision := configs.StubConfigs.Servers.Type["image"][server].Precision
		allclass := [][]string{{"bankcard_positive", "bankcard_negative"}, {"gongzhang", "gongzhang"}, {"idcard_negative"}, {"idcard_positive", "bankcard_positive"}, {"normal"}, {"mobile-screenshot"}, {"other-text"}, {"blog"}, {"blog"}}
		indexs := [][]int{{3, 4}, {5, 5}, {2}, {1, 3}, {33}, {49}, {45}, {28}, {28}}
		score := [][]float64{{0.999638, 0.999883}, {0.999035, 0.9988637}, {0.993002}, {0.9992399, 0.8430824}, {0.838102}, {0.999919}, {0.975641}, {0.946627}, {0.999897}}
		var set = "ocr-classify-argus/20181122/"
		var id = 0
		Context("tsv文件验证ocr/yinchuan", func() {
			for i := 0; i < len(imgs); i++ {
				uri := env.Env.GetURIPrivate(set + imgs[i])
				By("uri:" + uri)
				if err := biz.StoreUri(set+imgs[i], uri); err != nil {
					panic(err)
				}
				It("测试图片:"+imgs[i], func() {
					resp, err := client.PostWithJson(path, proto.NewArgusReq(uri, nil, nil))
					By("检查api请求:")
					Expect(err).Should(BeNil())
					By("检验api http code")
					Expect(resp.Status()).To(Equal(200))
					var resObj proto.ArgusCommonRes
					if err := json.Unmarshal(resp.BodyToByte(), &resObj); err != nil {
						Expect(err).Should(BeNil())
					}
					json_result, err := json.Marshal(resObj.Result)
					var results []Result
					Expect(json.Unmarshal(json_result, &results)).Should(BeNil())
					for j, result := range results {
						fmt.Println(result)
						fmt.Println(id, j)
						fmt.Println(indexs[id][j])
						Expect(result.Index).To(Equal(indexs[id][j]))
						Expect(result.Score).To(BeNumerically("~", score[id][j], precision))
						Expect(result.Class).To(Equal(allclass[id][j]))
						assert.CheckOcrPts(result.Bboxes)
					}
					id++
				})
			}
		})
	})
})

// Result ... result of expected response
type Result struct {
	Bboxes [4][2]int `json:"bboxes"`
	Class  string    `json:"class"`
	Index  int       `json:"index"`
	Score  float32   `json:"score"`
}

// Response ... response
type Response struct {
	Code    int      `json:"code"`
	Message string   `json:"message"`
	Result  []Result `json:"result"`
}
