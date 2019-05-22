package ops

import (
	"context"
	"encoding/json"
	"strconv"

	"github.com/qiniu/rpc.v3"
	"github.com/qiniu/xlog.v1"

	"qiniu.com/argus/video"
)

func RegisterTerror() {
	video.RegisterOP("terror",
		func(config video.OPConfig) video.OP { return NewSimpleCutOP2(config, NewEvalTerror, nil) })
}

type TerrorResp struct {
	Code    int          `json:"code"`
	Message string       `json:"message"`
	Result  TerrorResult `json:"result"`
}

type TerrorResult struct {
	Label  int     `json:"label"`
	Class  string  `json:"class,omitempty"`
	Score  float32 `json:"score"`
	Review bool    `json:"review"`
}

func (r TerrorResult) Len() int { return 1 }
func (r TerrorResult) Parse(i int) (string, float32, bool) {
	return strconv.Itoa(r.Label), r.Score, true
}

func NewEvalTerror(params video.OPParams) CutEval {
	var _params struct {
		Detail bool `json:"detail"`
	}
	{
		bs, _ := json.Marshal(params.Other)
		_ = json.Unmarshal(bs, &_params)
	}

	return func(ctx context.Context, client *rpc.Client, host, uri string) (interface{}, error) {
		var (
			xl   = xlog.FromContextSafe(ctx)
			resp TerrorResp
		)
		err := client.CallWithJson(ctx, &resp, "POST", host+"/v1/terror",
			struct {
				Data struct {
					URI string `json:"uri"`
				} `json:"data"`
				Params struct {
					Detail bool `json:"detail"`
				} `json:"params"`
			}{
				Data: struct {
					URI string `json:"uri"`
				}{
					URI: uri,
				},
				Params: struct {
					Detail bool `json:"detail"`
				}{
					Detail: _params.Detail,
				},
			},
		)
		if err != nil {
			return nil, err
		}
		if resp.Code != 0 && resp.Code/100 != 2 {
			xl.Warnf("terror cut failed. %#v", resp)
			return nil, nil
		}
		return resp.Result, nil
	}
}
