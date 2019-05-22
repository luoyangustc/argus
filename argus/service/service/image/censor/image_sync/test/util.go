package test

import (
	"encoding/json"
	"fmt"

	. "github.com/onsi/gomega"
	"qiniu.com/argus/test/biz/assert"
	"qiniu.com/argus/test/biz/client"
	"qiniu.com/argus/test/biz/proto"
)

// ################################### FOP BEGIN ##################################

// Response ...
type CensorResult struct {
	Suggestion string                     `json:"suggestion"`
	Scenes     map[string]json.RawMessage `json:"scenes"`
}

// Error ... image-censor error
type Error struct {
	Error string `json:"error"`
}

//Bounding_box
type BoundingBox struct {
	Pts   [][2]int `json:"pts,omitempty"`
	Score float64  `json:"score,omitempty"`
}

//Scenes ... scenes for response
type Scenes struct {
	Suggestion string `json:"suggestion"`
	Details    []struct {
		Suggestion string        `json:"suggestion"`
		Lable      string        `json:"label"`
		Group      string        `json:"group,omitempty"`
		Score      float64       `json:"score"`
		Detections []BoundingBox `json:"detections,omitempty"`
	} `json:"details"`
}

// ################################### FOP END ####################################

// ################################## CENSOR BEGIN ################################

// CensorData ... request data of censor api
type CensorData struct {
	URI string `json:"uri"`
}

// CensorReq .... censor api request
type CensorReq struct {
	Data   CensorData  `json:"data"`
	Params ParamsSenes `json:"params"`
}

type ParamsSenes struct {
	Scenes []string `json:"scenes"`
}

// GetCensorResp ... get censor response
func GetCensorResp(client client.Client, path string, scenes []string, fileUris string) ([]byte, error) {
	req := NewCensorReq(NewCensorDatas(fileUris), scenes)
	resp, err := client.PostWithJson(path, req)
	if err != nil {
		panic(err)
	}
	return resp.BodyToByte(), err
}

// NewCensorDatas ... new censor datas
func NewCensorDatas(uriArr string) CensorData {
	var rt CensorData
	rt.URI = uriArr
	return rt
}

// NewCensorReq ... new censor request
func NewCensorReq(data CensorData, scenes []string) CensorReq {
	var rt CensorReq
	rt.Data = data
	rt.Params.Scenes = scenes
	return rt
}

// ################################## CENSOR END ##################################

// ################################## FUNC  BEGIN #################################

func CheckAdsLabeDetail(scene Scenes, label string, suggestion string, group string) {
	if len(scene.Details) == 0 {
		return
	}
	MAXsocre := 0.0
	bExist := false
	for _, detail := range scene.Details {
		if detail.Lable == label {
			Expect(detail.Suggestion).To(Equal(suggestion))
			bExist = true
		}
		if len(detail.Detections) != 0 {
			for _, detection := range detail.Detections {
				if detection.Score >= MAXsocre {
					MAXsocre = detection.Score
				}
			}
			Expect(detail.Score).Should(Equal(MAXsocre))
		}
	}
	Expect(bExist).Should(Equal(true))
}
func CheckPulp(fopResp []byte, lable string, suggestion string) {
	var fopResponse proto.ArgusResponse //Response
	var Result CensorResult
	if err := json.Unmarshal(fopResp, &fopResponse); err != nil {
		panic(err)
	}
	//code
	Expect(fopResponse.Code).To(Equal(200))
	// 最外层suggestion
	res_byte, err := json.Marshal(fopResponse.Result)
	if err != nil {
		panic(err)
	}
	if err := json.Unmarshal(res_byte, &Result); err != nil {
		panic(err)
	}
	Expect(Result.Suggestion).To(Equal(suggestion))

	var fopScenes Scenes
	resp, reserr := json.Marshal(Result.Scenes["pulp"])
	Expect(reserr).Should(BeNil())
	if err := json.Unmarshal(resp, &fopScenes); err != nil {
		panic(err)
	}
	fmt.Printf("CheckPulp: fop scenes:%+v\n", fopScenes)
	// 里层suggestion
	Expect(fopScenes.Suggestion).To(Equal(suggestion))
	Expect(fopScenes.Details[0].Lable).To(Equal(lable))
}

// CheckPolitician ... group: domestic_statesman|foreign_statesman|affairs_official_gov|affairs_official_ent|anti_china_people|terrorist|affairs_celebrity
func CheckPolitician(fopResp []byte, group string, suggestion string, label string) Scenes {
	var fopResponse proto.ArgusResponse //Response
	var Result CensorResult
	if err := json.Unmarshal(fopResp, &fopResponse); err != nil {
		panic(err)
	}
	//code
	Expect(fopResponse.Code).To(Equal(200))
	// 最外层suggestion
	res_byte, err := json.Marshal(fopResponse.Result)
	if err != nil {
		panic(err)
	}
	if err := json.Unmarshal(res_byte, &Result); err != nil {
		panic(err)
	}
	Expect(Result.Suggestion).To(Equal(suggestion))

	var fopScenes Scenes
	resp, reserr := json.Marshal(Result.Scenes["politician"])
	Expect(reserr).Should(BeNil())
	if err := json.Unmarshal(resp, &fopScenes); err != nil {
		panic(err)
	}
	fmt.Printf("CheckPolitician: fop scenes:%+v\n", fopScenes)
	// 里层suggestion
	Expect(fopScenes.Suggestion).To(Equal(suggestion))

	if group == "normal" {
		Expect(fopScenes.Suggestion).To(Equal("pass"))
	} else {
		Expect(fopScenes.Suggestion).To(Equal(suggestion))
		Expect(fopScenes.Details[0].Lable).To(Equal(label))
		Expect(fopScenes.Details[0].Group).To(Equal(group))
		// 验证坐标合法性
		assert.CheckPts(fopScenes.Details[0].Detections[0].Pts)
	}

	return fopScenes
}

// CheckTerror ... label: normal|bloodiness|bomb... innerLabel:审核结果的label
func CheckTerror(fopResp []byte, label string, innerLabel string, suggestion string) {
	var fopResponse proto.ArgusResponse //Response
	var Result CensorResult
	if err := json.Unmarshal(fopResp, &fopResponse); err != nil {
		panic(err)
	}
	//code
	Expect(fopResponse.Code).To(Equal(200))
	// 最外层suggestion
	res_byte, err := json.Marshal(fopResponse.Result)
	if err != nil {
		panic(err)
	}
	if err := json.Unmarshal(res_byte, &Result); err != nil {
		panic(err)
	}
	Expect(Result.Suggestion).To(Equal(suggestion))

	var fopScenes Scenes
	resp, reserr := json.Marshal(Result.Scenes["terror"])
	Expect(reserr).Should(BeNil())
	if err := json.Unmarshal(resp, &fopScenes); err != nil {
		panic(err)
	}
	fmt.Printf("CheckTerror: fop scenes:%+v\n", fopScenes)
	// 里层suggestion
	Expect(fopScenes.Suggestion).To(Equal(suggestion))
	if len(fopScenes.Details) > 1 {
		Expect(fopScenes.Details[1].Lable).To(Equal(innerLabel))
		assert.CheckPts(fopScenes.Details[1].Detections[0].Pts)
	}
}

//CheckAds ...
func CheckAds(fopResp []byte, label string, suggestion string) {
	var fopResponse proto.ArgusResponse //Response
	var Result CensorResult
	if err := json.Unmarshal(fopResp, &fopResponse); err != nil {
		panic(err)
	}
	//code
	Expect(fopResponse.Code).To(Equal(200))
	// 最外层suggestion
	res_byte, err := json.Marshal(fopResponse.Result)
	if err != nil {
		panic(err)
	}
	if err := json.Unmarshal(res_byte, &Result); err != nil {
		panic(err)
	}
	Expect(Result.Suggestion).To(Equal(suggestion))

	var fopScenes Scenes
	resp, reserr := json.Marshal(Result.Scenes["ads"])
	Expect(reserr).Should(BeNil())
	if err := json.Unmarshal(resp, &fopScenes); err != nil {
		panic(err)
	}
	fmt.Printf("CheckADS: fop scenes:%+v\n", fopScenes)
	// 里层suggestion
	Expect(fopScenes.Suggestion).Should(Equal(suggestion))
	if suggestion != "pass" {
		CheckAdsLabeDetail(fopScenes, label, suggestion, "")
	}
}

// ################################## FUNC   END  #################################
