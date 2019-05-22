// Code generated by make gen.
// source: ./platform/src/qiniu.com/argus/argus/docs/inference_api/detection.md
// DO NOT EDIT!

package client

type DetectionResp struct {
	Code    int64              `json:"code"`
	Result  DetectionResp_sub1 `json:"result"`
	Message string             `json:"message"`
}

type DetectionResp_sub1 struct {
	Detections []DetectionResp_sub2 `json:"detections"`
}

type DetectionResp_sub2 struct {
	Class string    `json:"class"`
	Index int       `json:"index"`
	Pts   [][]int64 `json:"pts"`
	Score float64   `json:"score"`
}
