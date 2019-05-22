// Code generated by make gen.
// source: ./platform/src/qiniu.com/argus/argus/docs/inference_api/facex.md
// DO NOT EDIT!

package client

type FacexFeatureResp struct {
	Code         int64                   `json:"code"`
	FacexFeature []FacexFeatureResp_sub2 `json:"facex_feature"`
	Message      string                  `json:"message"`
}

type FacexFeatureResp_sub2 struct {
	Code    int64                   `json:"code"`
	Faces   []FacexFeatureResp_sub1 `json:"faces"`
	Message string                  `json:"message"`
	Name    string                  `json:"name"`
}

type FacexFeatureResp_sub1 struct {
	Code    int64     `json:"code"`
	Feature []float64 `json:"feature"`
	Message string    `json:"message"`
	Pts     [][]int64 `json:"pts"`
}