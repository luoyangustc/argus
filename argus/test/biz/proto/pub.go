package proto

import (
	"encoding/json"
)

type DataObject struct {
	URI       string      `json:"uri"`
	Attribute interface{} `json:"attribute,omitempty"`
}

type ArgusReq struct {
	Data struct {
		URI       string      `json:"uri"`
		Attribute interface{} `json:"attribute,omitempty"`
	} `json:"data"`
	Params interface{} `json:"params,omitempty"`
}

type ArgusBatchReq struct {
	Data   []DataObject `json:"data"`
	Params interface{}  `json:"params,omitempty"`
}

func NewArgusBatchReq(uri string, attribute, params interface{}) ArgusBatchReq {
	var req ArgusBatchReq
	var data DataObject
	data.URI = uri
	data.Attribute = attribute
	req.Data = []DataObject{data}
	req.Params = params
	return req
}

func (r *ArgusBatchReq) Add(uri string, attribute interface{}) *ArgusBatchReq {
	var data DataObject
	data.URI = uri
	data.Attribute = attribute
	r.Data = append(r.Data, data)
	return r
}

func NewArgusReq(uri string, attribute, params interface{}) ArgusReq {
	var req ArgusReq
	req.Data.URI = uri
	req.Data.Attribute = attribute
	req.Params = params
	return req
}

type ErrMessage struct {
	Message string `json:"error"`
	Code    int    `json:"code,omitempty"`
}

type Params struct {
	Limit int `json:"limit,omitempty"`
}

func NewLimit(limit int) Params {
	var params Params
	params.Limit = limit
	return params
}

func NewAttr(attrStr string) (interface{}, error) {
	var attr interface{}
	if err := json.Unmarshal([]byte(attrStr), &attr); err != nil {
		return attr, err
	}
	return attr, nil
}

type PtsObj [][2]int

func NewPtsAttr(attrStr string) (interface{}, error) {
	var pts PtsObj
	if err := json.Unmarshal([]byte(attrStr), &pts); err != nil {
		return pts, err
	}
	var attr = map[string]PtsObj{"pts": pts}
	return attr, nil
}

type ArgusRes struct {
	Code    int         `json:"code"`
	Message string      `json:"message"`
	Result  ArgusResult `json:"result"`
}

type ArgusResult struct {
	Label       int               `json:"label,omitempty"`
	Score       float64           `json:"score,omitempty"`
	Review      bool              `json:"review,omitempty"`
	Class       string            `json:"class,omitempty"`
	Confidences []ArgusConfidence `json:"confidences,omitempty"`
	Detections  []ArgusDetection  `json:"detections,omitempty"`
	Details     []ArgusDetails    `json:"details,omitempty"`
	Checkpoint  string            `json:"checkpoint"`
}

type ArgusResInter struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Result  struct {
		Label       int         `json:"label,omitempty"`
		Score       float64     `json:"score,omitempty"`
		Review      bool        `json:"review,omitempty"`
		Class       string      `json:"class,omitempty"`
		Confidences interface{} `json:"confidences,omitempty"`
		Detections  interface{} `json:"detections,omitempty"`
		Details     interface{} `json:"details,omitempty"`
		Checkpoint  string      `json:"checkpoint"`
	} `json:"result"`
}

type ArgusConfidence struct {
	Class  string       `json:"class,omitempty"`
	Group  string       `json:"group,omitempty"`
	Index  int          `json:"index,omitempty"`
	Score  float64      `json:"score,omitempty"`
	Sample ResultSample `json:"sample,omitempty"`
}

type ResultSample struct {
	Id  string   `json:"id,omitempty"`
	Pts [][2]int `json:"pts,omitempty"`
	Url string   `json:"url"`
}

type ArgusDetection struct {
	Class       string    `json:"class,omitempty"`
	Index       int       `json:"index,omitempty"`
	Orientation string    `json:"orientation,omitempty"`
	Quality     string    `json:"quality,omitempty"`
	Score       float64   `json:"score,omitempty"`
	Pts         [][2]int  `json:"pts,omitempty"`
	AreaRatio   float64   `json:"area_ratio,omitempty"`
	Qscore      QscoreObj `json:"q_score,omitempty"`
}

type QscoreObj struct {
	Blur  float64 `json:"blur"`
	Clear float64 `json:"clear"`
	Cover float64 `json:"cover"`
	Neg   float64 `json:"neg"`
	Pose  float64 `json:"pose"`
}

type ArgusDetails struct {
	Type   string  `json:"type"`
	Label  int     `json:"label"`
	Score  float64 `json:"score"`
	Review bool    `json:"review"`
}

type ArgusCommonRes struct {
	Code    int         `json:"code"`
	Message string      `json:"message"`
	Result  interface{} `json:"result"`
}

type ArgusResponse struct {
	Code    int         `json:"code"`
	Message string      `json:"message"`
	Result  interface{} `json:"result"`
}

type CodeErrorResp struct {
	Code  int    `json:"code"`
	Error string `json:"error"`
}
