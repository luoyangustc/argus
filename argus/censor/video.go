package censor

import (
	"context"
	"encoding/json"
	"time"

	httputil "github.com/qiniu/http/httputil.v1"
	"github.com/qiniu/http/restrpc.v1"
	"github.com/qiniu/rpc.v3"
	"github.com/qiniu/xlog.v1"
	authstub "qiniu.com/auth/authstub.v1"

	ahttp "qiniu.com/argus/argus/com/http"
	"qiniu.com/argus/argus/com/util"
	. "qiniu.com/argus/censor/biz"
	"qiniu.com/argus/video"
	"qiniu.com/argus/video/vframe"
)

type VideoConfig struct {
	NotifyURL string `json:"notify_url"`
}

type VideoRequest struct {
	Datas []struct {
		DataID string `json:"data_id,omitempty"`
		URI    string `json:"uri"` // TODO 统一URI定义
	} `json:"datas"`
	Scenes []Scene `json:"scenes,omitempty"`
	Params struct {
		Scenes  map[Scene]json.RawMessage `json:"scenes"`
		Vframe  *vframe.VframeParams      `json:"vframe"`
		Save    json.RawMessage           `json:"save,omitempty"`
		HookURL string                    `json:"hook_url"`
	} `json:"params,omitempty"`
}

type VideoResp struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	TaskID  string `json:"task_id"`
}

func (s Service) PostVideoAsyncrecognition(
	ctx context.Context, req *VideoRequest, env *authstub.Env,
) (ret struct {
	Tasks []VideoResp `json:"tasks"`
}, err error) {

	ctx, xl := util.CtxAndLog(ctx, env.W, env.Req)
	xl.Infof("post image ... %#v", req)

	// TODO 加监控

	var uid, utype uint32 = env.Uid, env.Utype
	_, _ = uid, utype

	if len(req.Scenes) == 0 {
		req.Scenes = DefaultScenes
	}

	reqs := formatAsyncVideoReqs(*req, s.VideoConfig)
	ret.Tasks = make([]VideoResp, 0, len(reqs))
	for _, req := range reqs {
		resp0, err := s.NewAsyncVideoClient(uid, utype).Post(ctx, req)
		if err != nil {
			code, msg := httputil.DetectError(err)
			ret.Tasks = append(ret.Tasks, VideoResp{Code: code, Message: msg})
		} else {
			ret.Tasks = append(ret.Tasks, VideoResp{Code: 200, Message: "OK", TaskID: resp0.Job})
		}
	}
	return
}

func (s Service) GetVideoTaskresults(
	ctx context.Context,
	req *struct {
		Tasks []struct {
			TaskID string `json:"task_id"`
		} `json:"tasks"`
	},
	env *authstub.Env,
) (
	results []struct {
		TaskID string         `json:"task_id"`
		Result CensorResponse `json:"result"`
	},
	err error,
) {

	ctx, xl := util.CtxAndLog(ctx, env.W, env.Req)
	xl.Infof("get video taskresults ... %#v", req)

	// TODO 加监控

	var uid, utype uint32 = env.Uid, env.Utype
	var cli = s.NewAsyncVideoClient(uid, utype)
	for _, task := range req.Tasks {
		var ret = CensorResponse{Suggestion: PASS, Scenes: map[Scene]interface{}{}}
		job, err := cli.GetJob(ctx, task.TaskID)
		if err != nil {
			ret.Code, ret.Message = httputil.DetectError(err)
		} else {
			ret.Code, ret.Message = 200, "OK"
			var resp0 = map[string]struct {
				Result OriginVideoOPResult `json:"result,omitempty"`
			}{}
			_ = json.Unmarshal(job.Result, &resp0)
			for op, ret0 := range resp0 {
				switch op {
				case "pulp":
					var params = struct {
						BlockThreshold PulpThreshold `json:"block_threshold"`
					}{}
					ret1 := ParseOriginVideoOPResult(ret0.Result,
						func(cut OriginCutResult) CutResult {
							return ParseCutPulpResult(cut, params.BlockThreshold)
						})
					ret.Suggestion = ret.Suggestion.Update(ret1.Suggestion)
					ret.Scenes[PULP] = ret1
				case "terror":
					var params = struct {
						BlockThreshold TerrorThreshold `json:"block_threshold"`
					}{}
					ret1 := ParseOriginVideoOPResult(ret0.Result,
						func(cut OriginCutResult) CutResult {
							return ParseCutTerrorResult(cut, params.BlockThreshold)
						})
					ret.Suggestion = ret.Suggestion.Update(ret1.Suggestion)
					ret.Scenes[TERROR] = ret1
				case "politician":
					var params = struct {
						BlockThreshold PoliticianThreshold `json:"block_threshold"`
					}{}
					ret1 := ParseOriginVideoOPResult(ret0.Result,
						func(cut OriginCutResult) CutResult {
							return ParseCutPoliticianResult(cut, params.BlockThreshold)
						})
					ret.Suggestion = ret.Suggestion.Update(ret1.Suggestion)
					ret.Scenes[POLITICIAN] = ret1
				}
			}
		}
		results = append(results, struct {
			TaskID string         `json:"task_id"`
			Result CensorResponse `json:"result"`
		}{
			TaskID: task.TaskID,
			Result: ret,
		})
	}

	return results, nil

}

func (s Service) PostVideoAsyncnotify(
	ctx context.Context,
	req *struct {
		ID     string         `json:"id"`
		Meta   AsyncVideoMeta `json:"meta"`
		Result map[string]struct {
			Result OriginVideoOPResult `json:"result,omitempty"`
		} `json:"result,omitempty"`
	},
	env *restrpc.Env,
) error {

	ctx, xl := util.CtxAndLog(ctx, env.W, env.Req)
	xl.Infof("video async notify ... %#v", req)

	var ret = CensorResponse{Suggestion: PASS, Scenes: map[Scene]interface{}{}}
	for op, ret0 := range req.Result {
		switch op {
		case "pulp":
			var params = struct {
				BlockThreshold PulpThreshold `json:"block_threshold"`
			}{}
			if raw, ok := req.Meta.Scenes[PULP]; ok && raw != nil {
				_ = json.Unmarshal(raw, &params)
			}
			ret1 := ParseOriginVideoOPResult(ret0.Result,
				func(cut OriginCutResult) CutResult {
					return ParseCutPulpResult(cut, params.BlockThreshold)
				})
			ret.Suggestion = ret.Suggestion.Update(ret1.Suggestion)
			ret.Scenes[PULP] = ret1
		case "terror":
			var params = struct {
				BlockThreshold TerrorThreshold `json:"block_threshold"`
			}{}
			if raw, ok := req.Meta.Scenes[TERROR]; ok && raw != nil {
				_ = json.Unmarshal(raw, &params)
			}
			ret1 := ParseOriginVideoOPResult(ret0.Result,
				func(cut OriginCutResult) CutResult {
					return ParseCutTerrorResult(cut, params.BlockThreshold)
				})
			ret.Suggestion = ret.Suggestion.Update(ret1.Suggestion)
			ret.Scenes[TERROR] = ret1
		case "politician":
			var params = struct {
				BlockThreshold PoliticianThreshold `json:"block_threshold"`
			}{}
			if raw, ok := req.Meta.Scenes[POLITICIAN]; ok && raw != nil {
				_ = json.Unmarshal(raw, &params)
			}
			ret1 := ParseOriginVideoOPResult(ret0.Result,
				func(cut OriginCutResult) CutResult {
					return ParseCutPoliticianResult(cut, params.BlockThreshold)
				})
			ret.Suggestion = ret.Suggestion.Update(ret1.Suggestion)
			ret.Scenes[POLITICIAN] = ret1
		}
	}

	if len(req.Meta.HookURL) > 0 {
		err := rpc.DefaultClient.CallWithJson(ctx, nil, "POST", req.Meta.HookURL,
			struct {
				DataID string         `json:"data_id,omitempty"`
				Result CensorResponse `json:"result"`
			}{DataID: req.Meta.DataID, Result: ret},
		)
		xl.Infof("notify. %s %v", req.Meta.HookURL, err)
	}
	return nil
}

//----------------------------------------------------------------------------//

func formatAsyncVideoReqs(req VideoRequest, cfg VideoConfig) []video.VideoRequest {
	var reqs = make([]video.VideoRequest, 0, len(req.Datas))
	var scenes = req.Scenes
	if len(scenes) == 0 {
		scenes = DefaultScenes
	}
	for _, data := range req.Datas {
		req0 := video.VideoRequest{}
		req0.Data.URI = data.URI
		req0.Data.Attribute.Meta, _ = json.Marshal(AsyncVideoMeta{
			Scenes:  req.Params.Scenes,
			DataID:  data.DataID,
			HookURL: req.Params.HookURL,
		})
		req0.Params.HookURL = cfg.NotifyURL
		req0.Params.Async = true
		req0.Params.Vframe = req.Params.Vframe
		if req.Params.Save != nil {
			req0.Params.Save = &req.Params.Save
		}
		for _, scene := range scenes {
			req0.Ops = append(req0.Ops, struct {
				OP             string         `json:"op"`
				CutHookURL     string         `json:"cut_hook_url"`
				SegmentHookURL string         `json:"segment_hook_url"`
				HookURL        string         `json:"hookURL"`
				Params         video.OPParams `json:"params"`
			}{OP: string(scene)})
			switch scene {
			case POLITICIAN:
				req0.Ops[len(req0.Ops)-1].Params.Other = struct {
					All bool `json:"all"`
				}{All: true}
			}
		}
		reqs = append(reqs, req0)
	}
	return reqs
}

////////////////////////////////////////////////////////////////////////////////

type NewAsyncVideoClient func(uint32, uint32) AsyncVideoClient

type AsyncVideoClient interface {
	Post(context.Context, video.VideoRequest) (AsyncVideoResp, error)
	GetJob(context.Context, string) (video.Job, error)
}

type AsyncVideoMeta struct {
	Scenes  map[Scene]json.RawMessage `json:"scenes"`
	DataID  string                    `json:"data_id"`
	HookURL string                    `json:"hook_url"`
}

type AsyncVideoResp struct {
	Job string `json:"job"`
}

type asyncVideoClient struct {
	Host       string
	Timeout    time.Duration
	UID, Utype uint32
}

func NewAsyncVideoHTTPClient(host string, timeout time.Duration) NewAsyncVideoClient {
	return func(uid, utype uint32) AsyncVideoClient {
		return asyncVideoClient{
			Host: host, Timeout: timeout,
			UID: uid, Utype: utype,
		}
	}
}

func (cli asyncVideoClient) Post(
	ctx context.Context, req video.VideoRequest,
) (AsyncVideoResp, error) {
	var (
		resp   AsyncVideoResp
		client = ahttp.NewQiniuStubRPCClient(cli.UID, cli.Utype, cli.Timeout)
		url    = cli.Host + "/v1/video/" + xlog.GenReqId()
		f      = func(ctx context.Context) error {
			return client.CallWithJson(ctx, &resp, "POST", url, req)
		}
	)
	err := ahttp.CallWithRetry(ctx, []int{530}, []func(context.Context) error{f, f})
	return resp, err
}

func (cli asyncVideoClient) GetJob(
	ctx context.Context, jobID string,
) (video.Job, error) {
	var (
		resp   video.Job
		client = ahttp.NewQiniuStubRPCClient(cli.UID, cli.Utype, cli.Timeout)
		url    = cli.Host + "/v1/jobs/video/" + jobID
		f      = func(ctx context.Context) error {
			return client.Call(ctx, &resp, "GET", url)
		}
	)
	err := ahttp.CallWithRetry(ctx, []int{530}, []func(context.Context) error{f, f})
	return resp, err
}
