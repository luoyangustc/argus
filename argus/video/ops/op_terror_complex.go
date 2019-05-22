package ops

import (
	"context"
	"encoding/json"
	"strconv"

	"github.com/qiniu/rpc.v3"
	"github.com/qiniu/xlog.v1"

	"qiniu.com/argus/video"
)

func RegisterTerrorComplex() {
	video.RegisterOP("terror_complex",
		func(config video.OPConfig) video.OP { return NewSimpleCutOP2(config, NewEvalTerrorComplex, nil) })
}

type TerrorComplexResp struct {
	Code    int                 `json:"code"`
	Message string              `json:"message"`
	Result  TerrorComplexResult `json:"result"`
}

type TerrorComplexResult struct {
	Label   int `json:"label"`
	Classes []struct {
		Class string  `json:"class,omitempty"`
		Score float32 `json:"score,omitempty"`
	} `json:"classes,omitempty"`
	Score  float32 `json:"score"`
	Review bool    `json:"review"`
}

func (r TerrorComplexResult) Len() int { return 1 }
func (r TerrorComplexResult) Parse(i int) (string, float32, bool) {
	return strconv.Itoa(r.Label), r.Score, true
}

func NewEvalTerrorComplex(params video.OPParams) CutEval {
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
			resp TerrorComplexResp
		)
		err := client.CallWithJson(ctx, &resp, "POST", host+"/v1/terror/complex",
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
