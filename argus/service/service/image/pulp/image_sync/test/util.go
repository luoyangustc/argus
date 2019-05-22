package test

import (
	"encoding/json"
	"strconv"

	K "github.com/onsi/ginkgo"
	O "github.com/onsi/gomega"
	"qiniu.com/argus/test/biz/proto"
)

func CalcPulpScore(x float64) (s float64) {
	threshold := 0.89
	if x > threshold {
		s = 0.4*(x-threshold)/(1-threshold) + 0.6
	} else {
		s = 0.6 * x / threshold
	}
	return
}

func Check(respone []byte, exp []byte, precision float64) {
	//以下为验证结果
	var resObj proto.ArgusRes
	if err := json.Unmarshal(respone, &resObj); err != nil {
		O.Expect(err).Should(O.BeNil())
	}
	var expObj proto.ArgusRes
	if err := json.Unmarshal(exp, &expObj); err != nil {
		O.Expect(err).Should(O.BeNil())
	}
	K.By("检验class个数")
	O.Expect(len(resObj.Result.Confidences)).To(O.Equal(len(expObj.Result.Confidences)))
	O.Expect(resObj.Result.Label).To(O.Equal(expObj.Result.Confidences[0].Index))
	O.Expect(resObj.Result.Score).To(O.BeNumerically("~", expObj.Result.Confidences[0].Score, precision))
	preScore := 1.1
	for _, ResultRun := range resObj.Result.Confidences {
		K.By("检验index：" + strconv.Itoa(ResultRun.Index))
		K.By("检验排序")
		O.Expect(ResultRun.Score).To(O.BeNumerically("<=", preScore))
		preScore = ResultRun.Score
		for _, confidence := range expObj.Result.Confidences {
			if ResultRun.Index == confidence.Index {
				K.By("检验score")
				// fmt.Println(ResultRun.Score)
				// fmt.Println(CalcPulpScore(confidence.Score))
				O.Expect(ResultRun.Score).To(O.BeNumerically("~", confidence.Score, precision))
				K.By("检验Class")
				O.Expect(ResultRun.Class).To(O.Equal(confidence.Class))
				break
			}
		}
	}
}

func CheckMiddle(respone []byte, exp1 []byte, exp2 []byte, precision float64) {
	//以下为验证结果
	var resObj proto.ArgusRes
	if err := json.Unmarshal(respone, &resObj); err != nil {
		O.Expect(err).Should(O.BeNil())
	}
	var exp1Obj proto.ArgusRes
	if err := json.Unmarshal(exp1, &exp1Obj); err != nil {
		O.Expect(err).Should(O.BeNil())
	}
	var exp2Obj proto.ArgusRes
	if err := json.Unmarshal(exp2, &exp2Obj); err != nil {
		O.Expect(err).Should(O.BeNil())
	}
	// O.Expect(len(resObj.Result.Confidences)).To(O.Equal(len(exp1Obj.Result.Confidences)))
	O.Expect(resObj.Result.Label).To(O.Equal(exp1Obj.Result.Confidences[0].Index))
	O.Expect(resObj.Result.Score).To(O.BeNumerically("~", CalcPulpScore((exp1Obj.Result.Confidences[0].Score+exp2Obj.Result.Confidences[0].Score)/2), precision))

}

func CheckFirst(respone []byte, exp []byte, precision float64) {
	//以下为验证结果
	var resObj proto.ArgusRes
	if err := json.Unmarshal(respone, &resObj); err != nil {
		O.Expect(err).Should(O.BeNil())
	}
	var expObj proto.ArgusRes
	if err := json.Unmarshal(exp, &expObj); err != nil {
		O.Expect(err).Should(O.BeNil())
	}
	K.By("检验class个数")
	O.Expect(len(resObj.Result.Confidences)).To(O.Equal(len(expObj.Result.Confidences)))
	O.Expect(resObj.Result.Label).To(O.Equal(expObj.Result.Confidences[0].Index))
	O.Expect(resObj.Result.Score).To(O.BeNumerically("~", expObj.Result.Confidences[0].Score, precision))
	preScore := 1.1
	ResultRun := resObj.Result.Confidences[0]
	K.By("检验index：" + strconv.Itoa(ResultRun.Index))
	K.By("检验排序")
	O.Expect(ResultRun.Score).To(O.BeNumerically("<=", preScore))
	// preScore = ResultRun.Score
	for _, confidence := range expObj.Result.Confidences {
		if ResultRun.Index == confidence.Index {
			K.By("检验score")
			// fmt.Println(ResultRun.Score)
			// fmt.Println(CalcPulpScore(confidence.Score))
			O.Expect(ResultRun.Score).To(O.BeNumerically("~", confidence.Score, precision))
			K.By("检验Class")
			O.Expect(ResultRun.Class).To(O.Equal(confidence.Class))
			break
		}
	}
}
