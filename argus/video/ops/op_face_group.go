package ops

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/qiniu/rpc.v3"
	"github.com/qiniu/xlog.v1"

	"qiniu.com/argus/video"
)

func RegisterFaceGroupSearch() {
	video.RegisterOP("face_group_search",
		func(config video.OPConfig) video.OP { return NewSimpleCutOP2(config, NewEvalFaceGroupSearch, nil) })
}

type FaceGroupSearchResp struct {
	Code    int                   `json:"code"`
	Message string                `json:"message"`
	Result  FaceGroupSearchResult `json:"result"`
}
type FaceGroupSearchResult struct {
	Review     bool `json:"review"`
	Detections []struct {
		BoundingBox struct {
			Pts   [][2]int `json:"pts"`
			Score float32  `json:"score"`
		} `json:"boundingBox" bson:"boundingBox"`
		Value struct {
			Id     string  `json:"id"`
			Name   string  `json:"name,omitempty"`
			Score  float32 `json:"score"`
			Review bool    `json:"review"`
		} `json:"value"`
		Sample *struct {
			URL string   `json:"url"`
			Pts [][2]int `json:"pts"`
		} `json:"sample,omitempty"`
	} `json:"detections"`
}

func (r FaceGroupSearchResult) Len() int { return len(r.Detections) }
func (r FaceGroupSearchResult) Parse(i int) (string, float32, bool) {
	return r.Detections[i].Value.Name, r.Detections[i].Value.Score, true
}

func NewEvalFaceGroupSearch(params video.OPParams) CutEval {
	var _params struct {
		Group string `json:"group"`
	}
	{
		bs, _ := json.Marshal(params.Other)
		_ = json.Unmarshal(bs, &_params)
	}

	return func(ctx context.Context, client *rpc.Client, host, uri string) (interface{}, error) {
		var (
			xl   = xlog.FromContextSafe(ctx)
			resp FaceGroupSearchResp
		)
		err := client.CallWithJson(ctx, &resp, "POST",
			host+fmt.Sprintf("/v1/face/group/%s/search", _params.Group),
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
			xl.Warnf("face group search cut failed. %#v", resp)
			return nil, nil
		}
		return resp.Result, nil
	}
}
