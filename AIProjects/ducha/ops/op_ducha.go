package ops

import (
	"context"

	"github.com/qiniu/rpc.v3"

	"qiniu.com/argus/video"
	"qiniu.com/argus/video/ops"
)

func RegisterDucha() {
	video.RegisterOP("ducha_classify",
		func(config video.OPConfig) video.OP {
			return ops.NewSimpleCutOP2(config, NewDuchaClassify, nil)
		})
	video.RegisterOP("ducha_detect",
		func(config video.OPConfig) video.OP {
			return ops.NewSimpleCutOP2(config, NewDuchaDetect, nil)
		})
}

type DuchaResp struct {
	Code    int         `json:"code"`
	Message string      `json:"message"`
	Result  DuchaResult `json:"result"`
}
type DuchaResult struct {
	Detections []struct {
		Class string   `json:"class"`
		Index int      `json:"index"`
		Pts   [][2]int `json:"pts"`
		Score float32  `json:"score"`
	} `json:"detections"`
}

func (r DuchaResult) Len() int { return 1 }
func (r DuchaResult) Parse(i int) (string, float32, bool) {
	return "", 1.0, true
}

func NewDuchaClassify(params video.OPParams) ops.CutEval {
	return func(ctx context.Context, client *rpc.Client, host, uri string) (interface{}, error) {
		var (
			resp DuchaResp
		)
		req := struct {
			Data struct {
				URI string `json:"uri"`
			} `json:"data"`
		}{}
		req.Data.URI = uri
		err := client.CallWithJson(ctx, &resp, "POST", host+"/v1/eval/ducha-c", req)
		if err != nil {
			return nil, err
		}
		return resp.Result, nil
	}
}

func NewDuchaDetect(params video.OPParams) ops.CutEval {
	return func(ctx context.Context, client *rpc.Client, host, uri string) (interface{}, error) {
		var (
			resp DuchaResp
		)
		req := struct {
			Data struct {
				URI string `json:"uri"`
			} `json:"data"`
		}{}
		req.Data.URI = uri
		err := client.CallWithJson(ctx, &resp, "POST", host+"/v1/eval/ducha-d", req)
		if err != nil {
			return nil, err
		}
		return resp.Result, nil
	}
}
