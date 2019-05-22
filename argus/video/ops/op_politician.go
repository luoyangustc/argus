package ops

import (
	"context"

	"github.com/qiniu/rpc.v3"
	"github.com/qiniu/xlog.v1"

	"qiniu.com/argus/video"
)

func RegisterPolitician() {
	video.RegisterOP("politician",
		func(config video.OPConfig) video.OP {
			return NewSimpleCutOP2(config, func(video.OPParams) CutEval { return EvalPolitician }, nil)
		})
}

type PoliticianResp struct {
	Code    int              `json:"code"`
	Message string           `json:"message"`
	Result  PoliticianResult `json:"result"`
}

type PoliticianResult struct {
	Detections []struct {
		BoundingBox struct {
			Pts   [][2]int `json:"pts"`
			Score float32  `json:"score"`
		} `json:"boundingBox"`
		Value struct {
			Name   string  `json:"name,omitempty"`
			Group  string  `json:"group,omitempty"`
			Score  float32 `json:"score"`
			Review bool    `json:"review"`
		} `json:"value"`
		Sample *struct {
			URL string   `json:"url"`
			Pts [][2]int `json:"pts"`
		} `json:"sample,omitempty"`
	} `json:"detections"`
}

func (r PoliticianResult) Len() int { return len(r.Detections) }

func (r PoliticianResult) Parse(i int) (string, float32, bool) {
	return r.Detections[i].Value.Name, r.Detections[i].Value.Score, len(r.Detections[i].Value.Name) > 0
}

func EvalPolitician(ctx context.Context, client *rpc.Client, host, uri string) (interface{}, error) {
	var (
		xl   = xlog.FromContextSafe(ctx)
		resp PoliticianResp
	)
	err := client.CallWithJson(ctx, &resp, "POST", host+"/v1/face/search/politician",
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
		xl.Warnf("politician cut failed. %#v", resp)
		return nil, nil
	}
	return resp.Result, nil
}
