package parser

import (
	"context"
	"encoding/json"
	"net/http"

	xlog "github.com/qiniu/xlog.v1"
	"qiniu.com/argus/AIProjects/wangan/image/wangan_mix"
	"qiniu.com/argus/AIProjects/wangan/yuqing"
	rpc "qiniupkg.com/x/rpc.v7"
)

type wanganHandler struct {
	Op        string
	ImageHost string
}

func NewWanganHandler(host string) OpHandler {
	return &wanganHandler{
		Op:        "wangan_mix",
		ImageHost: host,
	}
}

type WanganmixResult struct {
	Label   int      `json:"label"`
	Score   float32  `json:"score"`
	Classes []string `json:"classes"`
}

type ImageWanganRequest struct {
	Data struct {
		URI string `json:"uri"`
	} `json:"data"`
	Params struct {
		Detail bool `json:"detail"`
	} `json:"params"`
}

type ImageWanganResponse wangan_mix.WanganMixResp

func (h *wanganHandler) ImageParse(ctx context.Context, job yuqing.Job) (yuqing.Result, error) {

	var (
		req  ImageWanganRequest
		resp ImageWanganResponse
		ret  = yuqing.Result{
			Ops: make(map[string]yuqing.OpResult, 0),
		}
	)

	cli := rpc.Client{
		Client: &http.Client{
			Timeout: DEFAULT_TIMEOUT,
		},
	}

	url := h.ImageHost + "/v1/wangan-mix"
	req.Data.URI = job.URI
	req.Params.Detail = true
	err := cli.CallWithJson(ctx, &resp, "POST", url, req)
	if err != nil {
		xlog.FromContextSafe(ctx).Warnf("call wanga-mix image %s failed,req: %v, err: %v ", url, req, err)
		return yuqing.Result{}, err
	}

	if resp.Result.Label != 0 {
		var result yuqing.OpResult
		for _, class := range resp.Result.Classify {
			result.Labels = append(result.Labels, yuqing.OpResultLable{
				Name:  class.Class,
				Score: class.Score,
			})
		}
		for _, class := range resp.Result.Detection {
			result.Labels = append(result.Labels, yuqing.OpResultLable{
				Name:  class.Class,
				Score: class.Score,
			})
		}
		ret.Ops[h.Op] = result
		ret.Score = resp.Result.Score
	}

	return ret, nil
}

func (h *wanganHandler) GetVideoRequest() OpRequest {
	req := OpRequest{
		Op: h.Op,
	}
	req.Params.Labels = append(req.Params.Labels, struct {
		Label  string  `json:"label"`
		Select int     `json:"select"` // 0x01:INGORE; 0x02:ONLY
		Score  float32 `json:"score"`
	}{
		Label:  "normal",
		Select: 1,
	})
	req.Params.Other = struct {
		Detail bool   `json:"detail"`
		Type   string `json:"type"`
	}{
		Type: "internet_terror",
	}
	return req
}

func (h *wanganHandler) ParseVideoRespnse(resp OpResponse, uri string) yuqing.OpResult {
	var (
		ret yuqing.OpResult
	)

	ret.Labels = resp.Labels
	for _, segment := range resp.Segments {
		for _, cut := range segment.Cuts {

			var result WanganmixResult
			buf, _ := json.Marshal(cut.Result)
			_ = json.Unmarshal(buf, &result)
			ret.Cuts = append(ret.Cuts, yuqing.OpResultCut{
				//URI:     uri + fmt.Sprintf("?vframe/jpg/offset/%.3f", float32(cut.Offset)/float32(1000)),
				Offset:  float32(cut.Offset) / float32(1000),
				Score:   result.Score,
				Classes: result.Classes,
			})

		}
	}
	return ret
}
