package tsv

import (
	"bufio"
	"bytes"
	"encoding/json"
	"os"
	"regexp"
	"strconv"
	"strings"

	C "qiniu.com/argus/test/biz/client"
	"qiniu.com/argus/test/biz/env"
	"qiniu.com/argus/test/biz/proto"
	biz "qiniu.com/argus/test/biz/util"

	K "github.com/onsi/ginkgo"
	O "github.com/onsi/gomega"
)

func PickSetNum(value string) string {
	r, _ := regexp.Compile(`/set(\\d)*/`)
	pickValue := r.FindString(value)
	return strings.Replace(pickValue, "/", "", -1)
}

////////////////////////////////////////////////////////////////////////////////

func GetTestFile(file string) string {
	if name := os.Getenv("TEST_FILE"); name != "" {
		return name
	}
	return file
}

func GetImgSet(set string) string {
	if name := os.Getenv("TEST_IMAGE_SET"); name != "" {
		return name
	}
	return set
}

////////////////////////////////////////////////////////////////////////////////

type Tsv struct {
	Name      string
	Set       string
	Path      string
	Precision float64
	Params    interface{}
}

func NewTsv(file string, set string, path string, precision float64, params interface{}) Tsv {
	return Tsv{
		Name:      GetTestFile(file),
		Set:       GetImgSet(set),
		Path:      path,
		Precision: precision,
		Params:    params,
	}
}

func TsvTest(client C.Client, t Tsv, Check func([]byte, []byte, float64)) {
	buf, err := env.Env.GetTSV(t.Name)
	if err != nil {
		K.By("tsv下载失败：")
		panic(err)
	}
	// defer read.Close()
	reader := bufio.NewReader(bytes.NewReader(buf))
	// reader := bufio.NewReader(read)
	scanner := bufio.NewScanner(reader)
	scanner.Split(bufio.ScanLines)
	for scanner.Scan() {
		line := scanner.Text()
		record := strings.Split(line, "\t")
		K.By("tsv文件结果:\n" + line)
		uri := env.Env.GetURIPrivate(t.Set + record[0])
		if err := biz.StoreUri(t.Set+record[0], uri); err != nil {
			K.By(uri + " 下载失败：")
			panic(err)
		}
		K.It("测试图片:"+record[0], func() {
			//调用api获取最新结果
			resp, err := client.PostWithJson(t.Path,
				proto.NewArgusReq(uri, nil, t.Params))
			O.Expect(err).Should(O.BeNil())
			if resp.Status() == 200 {
				if Check == nil {
					return
				}
				Check(resp.BodyToByte(), []byte(record[1]), t.Precision)
			} else {
				CheckError(resp.BodyToByte(), []byte(record[1]))
			}
		})
	}
}

func TsvTestD(client C.Client, t Tsv, Check func([]byte, []byte, float64), Attr func(attrStr string) (interface{}, error)) {
	buf, err := env.Env.GetTSV(t.Name)
	if err != nil {
		K.By("tsv下载失败：")
		panic(err)
	}
	// defer read.Close()
	reader := bufio.NewReader(bytes.NewReader(buf))
	// reader := bufio.NewReader(read)
	scanner := bufio.NewScanner(reader)
	scanner.Split(bufio.ScanLines)
	for scanner.Scan() {
		line := scanner.Text()
		record := strings.Split(line, "\t")
		K.By("tsv文件结果:\n" + line)
		uri := env.Env.GetURIPrivate(t.Set + record[0])
		if err := biz.StoreUri(t.Set+record[0], uri); err != nil {
			K.By(uri + " 下载失败：")
			panic(err)
		}
		K.It("测试图片:"+record[0], func() {
			//调用api获取最新结果
			attr, err1 := Attr(record[1])
			O.Expect(err1).Should(O.BeNil())
			resp, err := client.PostWithJson(t.Path,
				proto.NewArgusReq(uri, attr, t.Params))
			O.Expect(err).Should(O.BeNil())
			if resp.Status() == 200 {
				if Check == nil {
					return
				}
				Check(resp.BodyToByte(), []byte(record[2]), t.Precision)
			} else {
				CheckError(resp.BodyToByte(), []byte(record[2]))
			}
		})
	}
}

func CheckClassify(respone []byte, exp []byte, precision float64) {
	//以下为验证结果
	var resObj proto.ArgusRes
	if err := json.Unmarshal(respone, &resObj); err != nil {
		O.Expect(err).Should(O.BeNil())
	}
	var expObj []proto.ArgusConfidence
	if err := json.Unmarshal(exp, &expObj); err != nil {
		O.Expect(err).Should(O.BeNil())
	}
	K.By("检验class个数")
	O.Expect(len(resObj.Result.Confidences)).To(O.Equal(len(expObj)))
	preScore := 1.1
	for _, ResultRun := range resObj.Result.Confidences {
		K.By("检验index：" + strconv.Itoa(ResultRun.Index))
		K.By("检验排序")
		O.Expect(ResultRun.Score).To(O.BeNumerically("<=", preScore))
		preScore = ResultRun.Score
		for _, confidence := range expObj {
			if ResultRun.Index == confidence.Index {
				K.By("检验score")
				O.Expect(ResultRun.Score).To(O.BeNumerically("~", confidence.Score, precision))
				K.By("检验Class")
				O.Expect(ResultRun.Class).To(O.Equal(confidence.Class))
				break
			}
		}
	}
}

func CheckClassifyEasy(respone []byte, exp []byte, precision float64) {
	//以下为验证结果
	var resObj proto.ArgusRes
	if err := json.Unmarshal(respone, &resObj); err != nil {
		O.Expect(err).Should(O.BeNil())
	}
	var expObj []proto.ArgusConfidence
	if err := json.Unmarshal(exp, &expObj); err != nil {
		O.Expect(err).Should(O.BeNil())
	}
	K.By("检验class个数")
	// O.Expect(len(resObj.Result.Confidences)).To(O.Equal(len(expObj)))
	preScore := 1.1
	for _, ResultRun := range resObj.Result.Confidences {
		K.By("检验index：" + strconv.Itoa(ResultRun.Index))
		K.By("检验排序")
		O.Expect(ResultRun.Score).To(O.BeNumerically("<=", preScore))
		preScore = ResultRun.Score
		for _, confidence := range expObj {
			if ResultRun.Index == confidence.Index {
				K.By("检验score")
				O.Expect(ResultRun.Score).To(O.BeNumerically("~", confidence.Score, precision))
				K.By("检验Class")
				O.Expect(ResultRun.Class).To(O.Equal(confidence.Class))
				break
			}
		}
	}
}

// CheckOcrClassify ... check ocr classify
func CheckOcrClassify(respone []byte, exp []byte, precision float64) {
	//以下为验证结果
	var resObj proto.ArgusRes
	if err := json.Unmarshal(respone, &resObj); err != nil {
		O.Expect(err).Should(O.BeNil())
	}
	var expObj proto.ArgusResult
	if err := json.Unmarshal(exp, &expObj); err != nil {
		O.Expect(err).Should(O.BeNil())
	}
	for kk, ResultRun := range resObj.Result.Confidences {
		K.By("===检验第" + strconv.Itoa(kk) + "个confidences数据===")
		K.By("检验Class")
		if expObj.Confidences[kk].Class == "other" {
			O.Expect(ResultRun.Class).To(O.Equal("normal"))
		} else {
			O.Expect(ResultRun.Class).To(O.Equal(expObj.Confidences[kk].Class))
		}
		K.By("检验Index")
		O.Expect(ResultRun.Index).To(O.Equal(expObj.Confidences[kk].Index))
		K.By("检验score")
		O.Expect(ResultRun.Score).To(O.BeNumerically("~", expObj.Confidences[kk].Score, precision))
	}
}

// CheckError ... check body
func CheckError(respone []byte, exp []byte) {
	//以下为验证结果
	var resObj proto.ErrMessage
	if err := json.Unmarshal(respone, &resObj); err != nil {
		O.Expect(err).Should(O.BeNil())
	}
	var expObj proto.ErrMessage
	if err := json.Unmarshal(exp, &expObj); err != nil {
		O.Expect(err).Should(O.BeNil())
	}
	O.Expect(resObj.Message).Should(O.Equal(expObj.Message))
	O.Expect(len(resObj.Message)).To(O.BeNumerically(">", 0))
	O.Expect(len(expObj.Message)).To(O.BeNumerically(">", 0))

}

// CheckImageTooLarge ... check big image
func CheckImageTooLarge(client C.Client, t Tsv, filepath string) {
	//以下为验证结果
	K.It("测试图片超过大小:"+filepath, func() {
		uri := env.Env.GetURIPrivate(filepath)
		resp, err := client.PostWithJson(t.Path, proto.NewArgusReq(uri, nil, t.Params))

		K.By("请检查服务是否正常，请求地址及路径是否正确")
		O.Expect(err).Should(O.BeNil())
		K.By("检验api返回http code")
		O.Expect(resp.Status()).To(O.Equal(400))

		// 以下为验证结果
		// var resObj proto.ErrMessage
		// if err := json.Unmarshal(resp.BodyToByte(), &resObj); err != nil {
		// 	O.Expect(err).Should(O.BeNil())
		// }
	})
}
