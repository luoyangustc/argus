package tsv

import (
	"encoding/json"
	"fmt"
	"strconv"

	K "github.com/onsi/ginkgo"
	O "github.com/onsi/gomega"
	"qiniu.com/argus/test/biz/proto"
)

func CheckCommon(respone []byte, expect []byte, precision float64) {
	//以下为验证结果
	var resObj proto.ArgusCommonRes
	if err := json.Unmarshal(respone, &resObj); err != nil {
		O.Expect(err).Should(O.BeNil())
	}
	var exp interface{}
	if err := json.Unmarshal(expect, &exp); err != nil {
		O.Expect(err).Should(O.BeNil())
	}
	CheckInterface(resObj.Result, exp, precision)
}

func CheckDetection(respone []byte, expect []byte, precision float64) {
	//以下为验证结果
	var resObj proto.ArgusResInter
	if err := json.Unmarshal(respone, &resObj); err != nil {
		O.Expect(err).Should(O.BeNil())
	}
	var exp interface{}
	if err := json.Unmarshal(expect, &exp); err != nil {
		O.Expect(err).Should(O.BeNil())
	}
	CheckInterface(resObj.Result.Detections, exp, precision)
}
func CheckInterface(act interface{}, exp interface{}, precision float64) {
	//以下为验证结果
	switch exp.(type) {
	case []interface{}:
		actA, ok := act.([]interface{})
		O.Expect(ok).Should(O.Equal(true))
		expA, ok := exp.([]interface{})
		O.Expect(ok).Should(O.Equal(true))
		CheckArrInterface(actA, expA, precision)
	case map[string]interface{}:
		actA, ok := act.(map[string]interface{})
		O.Expect(ok).Should(O.Equal(true))
		expA, ok := exp.(map[string]interface{})
		O.Expect(ok).Should(O.Equal(true))
		CheckStringInterface(actA, expA, precision)
	}
}

func CheckJson(respone []byte, expect []byte, precision float64) {
	//以下为验证结果
	var act interface{}
	if err := json.Unmarshal(respone, &act); err != nil {
		O.Expect(err).Should(O.BeNil())
	}
	var exp interface{}
	if err := json.Unmarshal(expect, &exp); err != nil {
		O.Expect(err).Should(O.BeNil())
	}
	switch exp.(type) {
	case []interface{}:
		actA, ok := act.([]interface{})
		O.Expect(ok).Should(O.Equal(true))
		expA, ok := exp.([]interface{})
		O.Expect(ok).Should(O.Equal(true))
		CheckArrInterface(actA, expA, precision)
	case map[string]interface{}:
		actA, ok := act.(map[string]interface{})
		O.Expect(ok).Should(O.Equal(true))
		expA, ok := exp.(map[string]interface{})
		O.Expect(ok).Should(O.Equal(true))
		CheckStringInterface(actA, expA, precision)
	}
}

func CheckStringInterface(act map[string]interface{}, exp map[string]interface{}, precision float64) {
	for k, v := range exp {
		K.By("校验：" + k)
		switch v.(type) {
		case string:
			O.Expect(act[k]).To(O.Equal(v))
		case int:
			O.Expect(act[k]).To(O.Equal(v))
		case bool:
			O.Expect(act[k]).To(O.Equal(v))
		case float64:
			O.Expect(act[k]).To(O.BeNumerically("~", v, precision))
		case []interface{}:
			actA, ok := act[k].([]interface{})
			O.Expect(ok).Should(O.Equal(true))
			expA, ok := exp[k].([]interface{})
			O.Expect(ok).Should(O.Equal(true))
			CheckArrInterface(actA, expA, precision)
		case map[string]interface{}:
			actA, ok := act[k].(map[string]interface{})
			O.Expect(ok).Should(O.Equal(true))
			expA, ok := exp[k].(map[string]interface{})
			O.Expect(ok).Should(O.Equal(true))
			CheckStringInterface(actA, expA, precision)
		default:
			fmt.Println(k, "is another type not handle yet")
		}
	}
}

func CheckArrInterface(act []interface{}, exp []interface{}, precision float64) {
	for k, v := range exp {
		K.By("校验index：" + strconv.Itoa(k))
		switch v.(type) {
		case string:
			O.Expect(act[k]).To(O.Equal(v))
		case int:
			O.Expect(act[k]).To(O.Equal(v))
		case bool:
			O.Expect(act[k]).To(O.Equal(v))
		case float64:
			O.Expect(act[k]).To(O.BeNumerically("~", v, precision))
		case []interface{}:
			actA, ok := act[k].([]interface{})
			O.Expect(ok).Should(O.Equal(true))
			expA, ok := exp[k].([]interface{})
			O.Expect(ok).Should(O.Equal(true))
			CheckArrInterface(actA, expA, precision)
		case map[string]interface{}:
			actA, ok := act[k].(map[string]interface{})
			O.Expect(ok).Should(O.Equal(true))
			expA, ok := exp[k].(map[string]interface{})
			O.Expect(ok).Should(O.Equal(true))
			CheckStringInterface(actA, expA, precision)
		default:
			fmt.Println(k, "is another type not handle yet")
		}
	}
}
