package ops

import (
	"context"
	"fmt"

	"github.com/qiniu/rpc.v3"
	"github.com/qiniu/xlog.v1"

	"qiniu.com/argus/video"
)

func RegisterTerrorDetect() {
	video.RegisterOP("terror_detect",
		func(config video.OPConfig) video.OP {
			return NewSimpleCutOP(config,
				func(ctx context.Context, client *rpc.Client, host, uri string) (interface{}, error) {
					return EvalTerrorDetect(ctx, client, host, "terror-detect", uri)
				},
			)
		})
	video.RegisterOP("terror_detect_t",
		func(config video.OPConfig) video.OP {
			return NewSimpleCutOP(config,
				func(ctx context.Context, client *rpc.Client, host, uri string) (interface{}, error) {
					return EvalTerrorDetect(ctx, client, host, "terror-detect-t", uri)
				},
			)
		})
}

type TerrorDetectResp struct {
	Code    int                `json:"code"`
	Message string             `json:"message"`
	Result  TerrorDetectResult `json:"result"`
}

type TerrorDetectResult struct {
	Detections []struct {
		Index int      `json:"index"`
		Class string   `json:"class"`
		Score float32  `json:"score"`
		Pts   [][2]int `json:"pts"`
	} `json:"detections"`
}

func (r TerrorDetectResult) Len() int { return len(r.Detections) }
func (r TerrorDetectResult) Parse(i int) (string, float32, bool) {
	return r.Detections[i].Class, r.Detections[i].Score, true
}

func EvalTerrorDetect(ctx context.Context, client *rpc.Client, host, op, uri string) (interface{}, error) {
	var (
		xl   = xlog.FromContextSafe(ctx)
		resp TerrorDetectResp
	)
	err := client.CallWithJson(ctx, &resp, "POST",
		host+fmt.Sprintf("/v1/eval/%s", op),
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
		xl.Warnf("terror detect cut failed. %#v", resp)
		return nil, nil
	}
	return resp.Result, nil
}
