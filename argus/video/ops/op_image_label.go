package ops

import (
	"context"

	"github.com/qiniu/rpc.v3"
	"github.com/qiniu/xlog.v1"

	"qiniu.com/argus/video"
)

func RegisterImageLabel() {
	video.RegisterOP("image_label",
		func(config video.OPConfig) video.OP { return NewSimpleCutOP(config, EvalImageLabel) })
}

type ImageLabelResp struct {
	Code    int              `json:"code"`
	Message string           `json:"message"`
	Result  ImageLabelResult `json:"result"`
}
type ImageLabelResult struct {
	Confidences []struct {
		Class string  `json:"class"`
		Score float32 `json:"score"`
	} `json:"confidences"`
}

func (r ImageLabelResult) Len() int { return len(r.Confidences) }
func (r ImageLabelResult) Parse(i int) (string, float32, bool) {
	return r.Confidences[i].Class, r.Confidences[i].Score, true
}

func EvalImageLabel(ctx context.Context, client *rpc.Client, host, uri string) (interface{}, error) {
	var (
		xl   = xlog.FromContextSafe(ctx)
		resp ImageLabelResp
	)
	err := client.CallWithJson(ctx, &resp, "POST", host+"/v1/image/label",
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
		xl.Warnf("image label cut failed. %#v", resp)
		return nil, nil
	}
	return resp.Result, nil
}
