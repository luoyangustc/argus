package ops

import (
	"context"

	"github.com/qiniu/rpc.v3"
	"github.com/qiniu/xlog.v1"

	"qiniu.com/argus/video"
)

func RegisterDetection() {
	video.RegisterOP("detection",
		func(config video.OPConfig) video.OP { return NewSimpleCutOP(config, EvalDetection) })
}

type DetectionResp struct {
	Code    int             `json:"code"`
	Message string          `json:"message"`
	Result  DetectionResult `json:"result"`
}
type DetectionResult struct {
	Detections []struct {
		Index int      `json:"index"`
		Class string   `json:"class"`
		Score float32  `json:"score"`
		Pts   [][2]int `json:"pts"`
	} `json:"detections"`
}

func (r DetectionResult) Len() int { return len(r.Detections) }
func (r DetectionResult) Parse(i int) (string, float32, bool) {
	return r.Detections[i].Class, r.Detections[i].Score, true
}

func EvalDetection(ctx context.Context, client *rpc.Client, host, uri string) (interface{}, error) {
	var (
		xl   = xlog.FromContextSafe(ctx)
		resp DetectionResp
	)
	err := client.CallWithJson(ctx, &resp, "POST", host+"/v1/eval/detection",
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
		xl.Warnf("detection cut failed. %#v", resp)
		return nil, nil
	}
	return resp.Result, nil
}
