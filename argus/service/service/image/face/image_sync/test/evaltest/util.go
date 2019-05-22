package evaltest

import (
	"encoding/json"
	"strconv"

	K "github.com/onsi/ginkgo"
	O "github.com/onsi/gomega"
	"qiniu.com/argus/test/biz/assert"
)

func CheckFacexFeature(respone []byte, exp []byte, precision float64) {
	//以下为验证结果
	var resObjtmp = make([]float32, 512)
	resObj := assert.ParseFloat32Buf(respone, resObjtmp)
	var expObj = make([]float32, 512)
	if err := json.Unmarshal(exp, &expObj); err != nil {
		O.Expect(err).Should(O.BeNil())
	}
	for i, point := range expObj {
		K.By("检验point: " + strconv.Itoa(i))
		// fmt.Println(point-resObj[i], resObj[i]-point)
		O.Expect((resObj[i]-point < float32(precision)) && (resObj[i]-point > 0-float32(precision))).To(O.Equal(true))
	}
}
