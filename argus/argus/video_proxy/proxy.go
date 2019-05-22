package video_proxy

import (
	"context"
	"strings"
	"time"

	"qiniu.com/argus/video"
	authstub "qiniu.com/auth/authstub.v1"
)

type ProxyReq struct {
	Mode     int      `json:"mode"`
	Interval int      `json:"interval"`
	URI      string   `json:"uri"` //Video data
	Cmds     []string `json:"cmds"`
	Limit    int      `json:"limit"` //快速鉴定限制，表示某种类别识别出敏感帧数到limit就返回，不继续鉴定，默认limit=0，表示鉴定完全视频
}

type Proxy interface {
	Post(context.Context, *ProxyReq, *authstub.Env) (interface{}, error)
}

type video_proxy struct {
	host string
}

func NewProxy(host string) video_proxy { return video_proxy{host: host} }

func (p video_proxy) PostVideo(ctx context.Context, req *ProxyReq, env *authstub.Env) (interface{}, error) {
	var (
		client = NewQiniuStubRPCClient(uint32(env.Uid), env.Utype, time.Second*60)
		ret    = new(map[string]struct {
			Labels   []video.ResultLabel   `json:"labels,omitempty"`
			Segments []video.SegmentResult `json:"segments"`
		})
	)

	videoReq := &video.VideoRequest{}
	videoReq.Data.URI = req.URI
	videoReq.CmdArgs = append(videoReq.CmdArgs, parseUri(req.URI))

	videoReq.Params.SegmentParams = &video.SegmentParams{}
	videoReq.Params.SegmentParams.Interval = req.Interval
	videoReq.Params.SegmentParams.Mode = req.Mode

	videoReq.Ops = make([]struct {
		OP             string         `json:"op"`
		CutHookURL     string         `json:"cut_hook_url"`
		SegmentHookURL string         `json:"segment_hook_url"`
		HookURL        string         `json:"hookURL"`
		Params         video.OPParams `json:"params"`
	}, len(req.Cmds))
	for i, cmd := range req.Cmds {
		videoReq.Ops[i].OP = cmd
	}

	err := client.CallWithJson(ctx, ret, "POST", p.host+"/v1/video/"+parseUri(req.URI), videoReq)

	type opResult struct {
		Offset      int64       `json:"offset,omitempty"`
		OffsetBegin int64       `json:"offset_begin,omitempty"`
		OffsetEnd   int64       `json:"offset_end,omitempty"`
		Result      interface{} `json:"result"`
		URI         string      `json:"uri,omitempty"`
	}
	type proxyResult struct {
		Result map[string][]opResult `json:"result"`
	}

	res := proxyResult{}
	res.Result = make(map[string][]opResult)
	for k, v := range *ret {
		var cutRes []opResult
		for _, val := range v.Segments {
			if len(val.Cuts) > 0 {
				for _, cut := range val.Cuts {
					cutRes = append(cutRes, opResult{
						Offset: cut.Offset,
						Result: cut.Result,
					})
				}
			} else if len(val.Clips) > 0 {
				for _, clip := range val.Clips {
					cutRes = append(cutRes, opResult{
						OffsetBegin: clip.OffsetBegin,
						OffsetEnd:   clip.OffsetEnd,
						Result:      clip.Result,
					})
				}
			}

		}
		res.Result[k] = cutRes
	}
	return res, err
}

func parseUri(str string) string {
	lines := strings.Split(str, "/")
	if len(lines) < 1 {
		return ""
	}

	nums := strings.Split(lines[len(lines)-1], ".")
	if len(nums) < 2 {
		return ""
	}
	return nums[0]
}
