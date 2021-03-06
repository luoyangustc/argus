// Code generated by make gen.
// source: ./platform/src/qiniu.com/argus/argus/docs/inference_api/facex.md
// DO NOT EDIT!

package client

type FacexFeatureReq2 struct {
	FacexDet []FacexFeatureReq2_sub2 `json:"facex_det"`
	Image    []string                `json:"image"`
}

type FacexFeatureReq2_sub1 struct {
	Class string    `json:"class"`
	Pts   [][]int64 `json:"pts"`
	Score float64   `json:"score"`
}

type FacexFeatureReq2_sub2 struct {
	Code       int64                   `json:"code"`
	Detections []FacexFeatureReq2_sub1 `json:"detections"`
	Message    string                  `json:"message"`
	Name       string                  `json:"name"`
}
