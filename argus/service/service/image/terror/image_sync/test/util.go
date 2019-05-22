package test

import (
	"encoding/json"

	. "github.com/onsi/gomega"
	"qiniu.com/argus/test/biz/proto"
)

func MaxScoreConfidences(confidences []proto.ArgusConfidence) proto.ArgusConfidence {
	idx := -1
	max := 0.0
	for i, confidence := range confidences {
		if confidence.Score >= max {
			idx = i
			max = confidence.Score
		}
	}
	return confidences[idx]
}

func IndexToLabel(index int) int {
	terror := []int{0, 4, 5, 6, 8, 9, 10, 11, 15, 17}
	for _, i := range terror {
		if index == i {
			return 1
		}
	}
	return 0
}

func DetectIndexToLabel(index int) int {

	if index == 12 {
		return 0
	}

	return 1
}

func CheckConfidences(checkpoint string, respone []byte, exp []byte, precision float64) {
	//以下为验证结果
	var resObj proto.ArgusRes
	if err := json.Unmarshal(respone, &resObj); err != nil {
		Expect(err).Should(BeNil())
	}
	var expObj proto.ArgusRes
	if err := json.Unmarshal(exp, &expObj); err != nil {
		Expect(err).Should(BeNil())
	}
	Expect(expObj.Result.Checkpoint).To(Equal(checkpoint))
	Expect(len(expObj.Result.Confidences)).To(BeNumerically(">", 0))
	expConfidence := MaxScoreConfidences(expObj.Result.Confidences)
	Expect(resObj.Result.Class).To(Equal(expConfidence.Class))
	Expect(resObj.Result.Score).To(BeNumerically("~", expConfidence.Score, precision))
	if resObj.Result.Class == "normal" {
		Expect(resObj.Result.Label).To(Equal(0))
	} else {
		Expect(resObj.Result.Label).To(Equal(1))
	}
}

func MaxScoreDetections(detections []proto.ArgusDetection) proto.ArgusDetection {
	idx := -1
	max := 0.0
	for i, detection := range detections {
		if detection.Score >= max {
			idx = i
			max = detection.Score
		}
	}
	return detections[idx]
}

func CheckDetections(respone []byte, exp []byte, precision float64) {
	//以下为验证结果
	var resObj proto.ArgusRes
	if err := json.Unmarshal(respone, &resObj); err != nil {
		Expect(err).Should(BeNil())
	}
	var expObj proto.ArgusRes
	if err := json.Unmarshal(exp, &expObj); err != nil {
		Expect(err).Should(BeNil())
	}
	// Expect(expObj.Result.Checkpoint).To(Equal("endpoint"))
	Expect(len(expObj.Result.Detections)).To(BeNumerically(">", 0))
	expDetection := MaxScoreDetections(expObj.Result.Detections)
	Expect(resObj.Result.Class).To(Equal(expDetection.Class))
	Expect(resObj.Result.Score).To(BeNumerically("~", expDetection.Score, precision))
	Expect(resObj.Result.Label).To(Equal(DetectIndexToLabel(expDetection.Index)))
}
