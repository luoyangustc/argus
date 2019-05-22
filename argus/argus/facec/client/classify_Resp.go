// Code generated by make gen.
// source: ./platform/src/qiniu.com/argus/argus/docs/inference_api/classify.md
// DO NOT EDIT!

package client

type SceneResp struct {
	Code    int64          `json:"code"`
	Message string         `json:"message"`
	Result  SceneResp_sub1 `json:"result"`
}

type SceneResp_sub1 struct {
	Confidences []SceneResp_sub2 `json:"confidences"`
}

type SceneResp_sub2 struct {
	Class string   `json:"class"`
	Index int      `json:"index"`
	Label []string `json:"label"`
	Score float64  `json:"score"`
}