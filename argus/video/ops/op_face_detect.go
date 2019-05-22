package ops

import (
	"context"

	"github.com/qiniu/rpc.v3"
	"github.com/qiniu/xlog.v1"

	"qiniu.com/argus/video"
)

func RegisterFaceDetect() {
	video.RegisterOP("face_detect",
		func(config video.OPConfig) video.OP { return NewSimpleCutOP(config, EvalFaceDetect) })
}

type FaceDetectResp struct {
	Code    int              `json:"code"`
	Message string           `json:"message"`
	Result  FaceDetectResult `json:"result"`
}
type FaceDetectResult struct {
	Detections []struct {
		BoundingBox struct {
			Pts   [][2]int `json:"pts"`
			Score float32  `json:"score"`
		} `json:"boundingBox"`
		Age struct {
			Value float32 `json:"value"`
			Score float32 `json:"score"`
		} `json:"age"`
		Gender struct {
			Value string  `json:"value"`
			Score float32 `json:"score"`
		} `json:"gender"`
	} `json:"detections"`
}

func (r FaceDetectResult) Len() int { return len(r.Detections) }
func (r FaceDetectResult) Parse(i int) (string, float32, bool) {
	return "", r.Detections[i].BoundingBox.Score, true
}

func EvalFaceDetect(ctx context.Context, client *rpc.Client, host, uri string) (interface{}, error) {
	var (
		xl   = xlog.FromContextSafe(ctx)
		resp FaceDetectResp
	)
	err := client.CallWithJson(ctx, &resp, "POST", host+"/v1/face/detect",
		struct {
			Data struct {
				URI string `json:"uri"`
			} `json:"data"`
		}{
			Data: struct {
				URI string `json:"uri"`
			}{
				URI: uri,
			},
		},
	)
	if err != nil {
		return nil, err
	}
	if resp.Code != 0 && resp.Code/100 != 2 {
		xl.Warnf("face detect cut failed. %#v", resp)
		return nil, nil
	}
	return resp.Result, nil
}
