// Code generated by make gen.
// source: ./platform/src/qiniu.com/argus/argus/docs/inference_api/detection.md
// DO NOT EDIT!

package client

type DetectionReq struct {
	Data DetectionReq_sub1 `json:"data"`
}

type DetectionReq_sub1 struct {
	URI string `json:"uri"`
}
