package gate

import (
	"context"
	"net/http"
	"time"

	xlog "github.com/qiniu/xlog.v1"
	imix "qiniu.com/argus/AIProjects/wangan/image/wangan_mix"
	"qiniu.com/argus/video"
	"qiniu.com/argus/video/vframe"
	rpc "qiniupkg.com/x/rpc.v7"
)

const (
	DEFAULT_TIMEOUT = 30000000000 // 30s
)

type Client interface {
	CallWithJson(context.Context, JsonRequest) (interface{}, error)
}

type ClientConfig struct {
	Host    string        `json:"host"`
	Timeout time.Duration `json:"timeout"`
}

type imageClient struct {
	ClientConfig
	*rpc.Client
}

func NewImageClient(conf ClientConfig) Client {
	if conf.Timeout == 0 {
		conf.Timeout = DEFAULT_TIMEOUT
	}
	return &imageClient{
		ClientConfig: conf,
		Client: &rpc.Client{
			Client: &http.Client{
				Timeout: conf.Timeout,
			},
		},
	}
}

func (c *imageClient) CallWithJson(ctx context.Context, req JsonRequest) (interface{}, error) {
	type WanganMixReq struct {
		Data struct {
			URI string `json:"uri"`
		} `json:"data"`
		Params struct {
			Detail bool   `json:"detail"`
			Type   string `json:"type"`
		} `json:"params"`
	}

	var (
		req1 WanganMixReq
		resp imix.WanganMixResp
	)

	req1.Data.URI = req.Data.URI
	req1.Params.Type = req.Params.Type
	url := c.Host + "/v1/wangan-mix"

	err := c.Client.CallWithJson(ctx, &resp, "POST", url, req1)
	if err != nil {
		return nil, err
	}
	return resp, nil
}

type videoClient struct {
	ClientConfig
	*rpc.Client
	interval float64
}

func NewVideoClient(conf ClientConfig, interval float64) Client {
	if conf.Timeout == 0 {
		conf.Timeout = DEFAULT_TIMEOUT
	}
	return &videoClient{
		ClientConfig: conf,
		Client: &rpc.Client{
			Client: &http.Client{
				Timeout: conf.Timeout,
			},
		},
		interval: interval,
	}
}

func (c *videoClient) CallWithJson(ctx context.Context, req JsonRequest) (interface{}, error) {
	type WanganMixResp struct {
		WanganMix struct {
			Labels   []video.ResultLabel   `json:"labels"`
			Segments []video.SegmentResult `json:"segments"`
		} `json:"wangan_mix"`
	}

	var (
		xl      = xlog.FromContextSafe(ctx)
		req1    video.VideoRequest
		resp    WanganMixResp
		mode    int = 0
		opParam video.OPParams
	)

	req1.Data.URI = req.Data.URI
	req1.Params.Vframe = &vframe.VframeParams{
		Mode:     &mode,
		Interval: c.interval,
	}
	opParam.Labels = append(opParam.Labels, struct {
		Name   string  `json:"label"`
		Select int     `json:"select"` // 0x01:INGORE; 0x02:ONLY
		Score  float32 `json:"score"`
	}{
		Name:   "normal",
		Select: 1,
	})
	opParam.Other = struct {
		Type string `json:"type"`
	}{
		Type: req.Params.Type,
	}
	req1.Ops = append(req1.Ops, struct {
		OP             string         `json:"op"`
		CutHookURL     string         `json:"cut_hook_url"`
		SegmentHookURL string         `json:"segment_hook_url"`
		HookURL        string         `json:"hookURL"`
		Params         video.OPParams `json:"params"`
	}{
		OP:     "wangan_mix",
		Params: opParam,
	})
	url := c.Host + "/v1/video/" + xl.ReqId()

	err := c.Client.CallWithJson(ctx, &resp, "POST", url, req1)
	if err != nil {
		xl.Warnf("video client call %s failed, req: %v, error: %s", url, req1, err.Error())
		return nil, err
	}
	return resp, nil
}
