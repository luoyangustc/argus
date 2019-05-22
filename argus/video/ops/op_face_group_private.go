package ops

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/qiniu/rpc.v3"

	feature_group "qiniu.com/argus/feature_group_private"
	"qiniu.com/argus/video"
)

func RegisterFaceGroupPrivateSearch() {
	video.RegisterOP("face_group_search_private",
		func(config video.OPConfig) video.OP {
			return NewSimpleCutOP2(config, NewEvalFaceGroupPrivateSearch, nil)
		})
}

type FaceGroupPrivateSearchResp struct {
	Results []FaceGroupPrivateSearchResult `json:"search_results"`
}
type FaceGroupPrivateSearchResult struct {
	Faces []feature_group.FaceSearchRespItem `json:"faces"`
}

func (r FaceGroupPrivateSearchResult) Len() int { return len(r.Faces) }
func (r FaceGroupPrivateSearchResult) Parse(i int) (string, float32, bool) {
	var (
		name  string
		score float32
	)

	// 选择TOP 1的人脸
	var ok bool
	if len(r.Faces[i].Faces) > 0 {
		name = string(r.Faces[i].Faces[0].Tag)
		score = r.Faces[i].Faces[0].Score
		ok = true
	}
	return name, score, ok
}

func NewEvalFaceGroupPrivateSearch(params video.OPParams) CutEval {
	var _params struct {
		Group     string  `json:"group"`
		Threshold float32 `json:"threshold"`
		Limit     int     `json:"limit"`
	}
	{
		bs, _ := json.Marshal(params.Other)
		_ = json.Unmarshal(bs, &_params)
	}

	return func(ctx context.Context, client *rpc.Client, host, uri string) (interface{}, error) {
		var (
			resp FaceGroupPrivateSearchResp
		)
		req := struct {
			Images    []string `json:"images"`
			Threshold float32  `json:"threshold"`
			Limit     int      `json:"limit"`
		}{
			Images:    []string{uri},
			Threshold: _params.Threshold,
			Limit:     _params.Limit,
		}
		err := client.CallWithJson(ctx, &resp, "POST",
			host+fmt.Sprintf("/v1/face/groups/%s/search", _params.Group),
			req,
		)
		if err != nil {
			return nil, err
		}
		return resp.Results[0], nil
	}
}
