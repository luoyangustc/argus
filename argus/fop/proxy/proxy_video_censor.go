package proxy

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"

	httputil "github.com/qiniu/http/httputil.v1"
	"github.com/qiniu/http/restrpc.v1"
	"github.com/qiniu/xlog.v1"

	ahttp "qiniu.com/argus/argus/com/http"
	"qiniu.com/argus/censor/biz"
	"qiniu.com/argus/com/proxy/fop"
	"qiniu.com/argus/video"
	"qiniu.com/argus/video/vframe"
)

var _ fop.Proxy = VideoCensor{}

type VideoCensor struct {
	URL string
}

func NewVideoCensor(_url string) fop.Proxy { return VideoCensor{URL: _url} }

func (p VideoCensor) Post(
	ctx context.Context, req fop.ProxyReq, env *restrpc.Env,
) (interface{}, error) {

	xl := xlog.FromContextSafe(ctx)

	uid, bucket, key, err := reqSourceInfo(env.Req)
	xl.Infof("SrcInfo, %d, %s, %s, %+v", uid, bucket, key, err)

	var (
		req2     video.VideoRequest
		saveInfo = struct {
			Bucket string `json:"bucket"`
			Prefix string `json:"prefix,omitempty"`
		}{}
		vframeInfo *vframe.VframeParams
	)

	if args := strings.Split(req.Cmd, "/"); len(args) > 1 {
		switch args[1] {
		case "v1":
			for i := 2; i < len(args); i++ {
				switch args[i] {
				case "pulp", "terror":
					req2.Ops = append(req2.Ops, struct {
						OP             string         `json:"op"`
						CutHookURL     string         `json:"cut_hook_url"`
						SegmentHookURL string         `json:"segment_hook_url"`
						HookURL        string         `json:"hookURL"`
						Params         video.OPParams `json:"params"`
					}{OP: args[i]})
				case "politician":
					req2.Ops = append(req2.Ops, struct {
						OP             string         `json:"op"`
						CutHookURL     string         `json:"cut_hook_url"`
						SegmentHookURL string         `json:"segment_hook_url"`
						HookURL        string         `json:"hookURL"`
						Params         video.OPParams `json:"params"`
					}{
						OP: args[i],
						Params: video.OPParams{Other: struct {
							All bool `json:"all"`
						}{All: true}},
					})
				case "save":
					if (i + 1) < len(args) {
						saveBytes, err := base64.StdEncoding.DecodeString(args[i+1])
						if err != nil {
							xl.Infof("base64 DecodeString err, %+v", err)
						} else {
							err := json.Unmarshal(saveBytes, &saveInfo)
							if err != nil {
								xl.Infof("json Unmarshal err, %+v", err)
							}
						}
					}
				case "vframe":
					if (i + 1) < len(args) {
						vframeBytes, err := base64.StdEncoding.DecodeString(args[i+1])
						if err != nil {
							xl.Infof("base64 DecodeString err, %+v", err)
						} else {
							vframeInfo = &vframe.VframeParams{}
							err := json.Unmarshal(vframeBytes, vframeInfo)
							if err != nil {
								xl.Infof("json Unmarshal err, %+v", err)
							}
						}
					}
				}
			}
		}
	}
	if len(req2.Ops) == 0 {
		req2.Ops = []struct {
			OP             string         `json:"op"`
			CutHookURL     string         `json:"cut_hook_url"`
			SegmentHookURL string         `json:"segment_hook_url"`
			HookURL        string         `json:"hookURL"`
			Params         video.OPParams `json:"params"`
		}{{OP: "pulp"}, {OP: "terror"},
			{
				OP: "politician",
				Params: video.OPParams{Other: struct {
					All bool `json:"all"`
				}{All: true}},
			}}
	}

	// 补全存帧信息
	if req2.Params.Save == nil {

		vss := struct {
			UID    uint32 `json:"uid,omitempty"`
			Zone   int    `json:"zone,omitempty"`
			Bucket string `json:"bucket"`
			Prefix string `json:"prefix,omitempty"`
		}{
			UID:    uid,
			Bucket: bucket,
			Prefix: "qiniu_censor_tmp/",
		}

		if saveInfo.Bucket != "" { // 允许自定义
			vss.Bucket = saveInfo.Bucket
			if saveInfo.Prefix != "" {
				vss.Prefix = saveInfo.Prefix + "/" + vss.Prefix
			}
		}

		vsRaw, _ := json.Marshal(vss)
		req2.Params.Save = (*json.RawMessage)(&vsRaw)
	}

	if req2.Params.Vframe == nil {
		req2.Params.Vframe = vframeInfo
	}

	req2.Data.URI = fmt.Sprintf("qiniu://%d@/%s/%s", req.UID, bucket, key)
	rstr, _ := json.Marshal(req2)
	xl.Infof("req: %s", string(rstr))

	var (
		client = ahttp.NewQiniuStubRPCClient(uint32(req.UID), 4, time.Second*3600)
		ret    = map[string]biz.OriginVideoOPResult{}
		url    = p.URL + "/" + xlog.GenReqId()
		call   = func(ctx context.Context) error {
			return client.CallWithJson(ctx, &ret, "POST", url, req2)
		}
	)

	xl.Infof("query video: %s", url)
	err = ahttp.CallWithRetry(ctx, []int{530}, []func(context.Context) error{call, call})
	if err != nil {
		xl.Errorf("ret, %s, %+v", JsonStr(ret), err)
		return ret, err
	}

	var ret2 = biz.CensorResponse{Suggestion: biz.PASS, Scenes: map[biz.Scene]interface{}{}}
	for op, ret0 := range ret {
		switch op {
		case "pulp":
			ret1 := biz.ParseOriginVideoOPResult(ret0, func(cut biz.OriginCutResult) biz.CutResult {
				return biz.ParseCutPulpResult(cut, biz.PulpThreshold{})
			})
			ret2.Suggestion = ret2.Suggestion.Update(ret1.Suggestion)
			ret2.Scenes[biz.PULP] = ret1
		case "terror":
			ret1 := biz.ParseOriginVideoOPResult(ret0, func(cut biz.OriginCutResult) biz.CutResult {
				return biz.ParseCutTerrorResult(cut, biz.TerrorThreshold{})
			})
			ret2.Suggestion = ret2.Suggestion.Update(ret1.Suggestion)
			ret2.Scenes[biz.TERROR] = ret1
		case "politician":
			ret1 := biz.ParseOriginVideoOPResult(ret0, func(cut biz.OriginCutResult) biz.CutResult {
				return biz.ParseCutPoliticianResult(cut, biz.PoliticianThreshold{})
			})
			ret2.Suggestion = ret2.Suggestion.Update(ret1.Suggestion)
			ret2.Scenes[biz.POLITICIAN] = ret1
		}
	}

	xl.Infof("ret, %s, %+v", JsonStr(ret2), err)
	return ret2, err
}

func reqSourceInfo(req *http.Request) (uint32, string, string, error) {
	var (
		uid    uint32
		bucket string
		key    string
	)
	{
		str := req.Header.Get("X-Qiniu-Uid")
		if str == "" {
			return uid, bucket, key, httputil.NewError(http.StatusBadRequest, "need params: uid")
		}
		bs, err := base64.StdEncoding.DecodeString(str)
		if err != nil {
			return uid, bucket, key, httputil.NewError(
				http.StatusBadRequest,
				fmt.Sprintf("bad params: uid, %s", str),
			)
		}
		uid64, err := strconv.ParseUint(string(bs), 10, 64)
		if err != nil {
			return uid, bucket, key, httputil.NewError(
				http.StatusBadRequest,
				fmt.Sprintf("bad params: uid, %s", string(bs)),
			)
		}
		uid = uint32(uid64)
	}

	{
		_bucket := req.Header.Get("X-Qiniu-Bucket")
		if _bucket == "" {
			return uid, bucket, key, httputil.NewError(http.StatusBadRequest, "need bucket param")
		}
		bucketBts, err := base64.StdEncoding.DecodeString(_bucket)
		if err != nil {
			return uid, bucket, key, httputil.NewError(http.StatusBadRequest, "decode bucket name error")
		}
		bucket = string(bucketBts)
	}
	{
		_key := req.Header.Get("X-Qiniu-Key")
		if _key == "" {
			return uid, bucket, key, httputil.NewError(http.StatusBadRequest, "need file key param")
		}
		keyBts, err := base64.StdEncoding.DecodeString(_key)
		if err != nil {
			return uid, bucket, key, httputil.NewError(http.StatusBadRequest, "decode file key error")
		}
		key = string(keyBts)
	}

	return uid, bucket, key, nil
}
