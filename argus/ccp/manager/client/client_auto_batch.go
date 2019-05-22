package client

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	xlog "github.com/qiniu/xlog.v1"
	ahttp "qiniu.com/argus/argus/com/http"
	"qiniu.com/argus/ccp/manager/proto"
	"qiniu.com/argus/ccp/manager/proto/kodo"
	"qiniu.com/argus/censor/biz"
	"qiniu.com/argus/video/vframe"
)

// call auto
type Bjobs interface {
	New(ctx context.Context, rule *proto.Rule) (string, error)
	Cancel(ctx context.Context, uid uint32, jobID string) error
}

var _ Bjobs = bJobs{}

type bJobs struct {
	InnerConfig
	Host string
}

func NewBJobs(innerCfg InnerConfig, host string) bJobs {
	return bJobs{
		InnerConfig: innerCfg,
		Host:        host,
	}
}

type _BJobsNewReq struct {
	UID    uint32 `json:"uid"`
	Utype  uint32 `json:"utype"`
	Zone   int    `json:"zone,omitempty"`
	Bucket string `json:"bucket"`
	Prefix string `json:"prefix,omitempty"`

	MimeTypes []string `json:"mimetypes,omitempty"`
	Params    struct {
		Image json.RawMessage `json:"image,omitempty"`
		Video json.RawMessage `json:"video,omitempty"`
	} `json:"params"`

	Save *Saver `json:"save,omitempty"`
}

func (b bJobs) genNewReq(ctx context.Context, rule *proto.Rule) (*_BJobsNewReq, error) {
	xl := xlog.FromContextSafe(ctx)

	kodoSrc, _, err := kodo.UnmarshalRule(ctx, rule)
	if err != nil {
		xl.Errorf("kodo.UnmarshalRule err, %+v", err)
		return nil, err
	}

	if len(kodoSrc.Buckets) <= 0 {
		err := fmt.Errorf("kodoSrc.Buckets empty, %+v", kodoSrc)
		xl.Errorf("%+v", err)
		return nil, err
	}

	req := _BJobsNewReq{
		UID:    rule.UID,
		Utype:  rule.Utype,
		Bucket: kodoSrc.Buckets[0].Bucket,

		Save: b.GetInnerSaver(ctx, fmt.Sprintf("bjob/%d/%s",
			rule.UID, kodoSrc.Buckets[0].Bucket)),
	}

	if rule.Image.IsOn {
		scenes := make([]biz.Scene, 0, len(rule.Image.Scenes))
		for _, ss := range rule.Image.Scenes {
			scenes = append(scenes, biz.Scene(ss))
		}
		req.MimeTypes = append(req.MimeTypes, "image")
		imgParamsRaw, _ := json.Marshal(struct {
			Scenes []biz.Scene `json:"scenes,omitempty"`
			Params struct {
				Scenes map[biz.Scene]json.RawMessage `json:"scenes"`
			} `json:"params,omitempty"`
		}{
			Scenes: scenes,
		})
		req.Params.Image = imgParamsRaw
	}
	if rule.Video.IsOn {
		scenes := make([]biz.Scene, 0, len(rule.Video.Scenes))
		for _, ss := range rule.Video.Scenes {
			scenes = append(scenes, biz.Scene(ss))
		}
		req.MimeTypes = append(req.MimeTypes, "video")
		vsaver := b.GetInnerSaver(ctx, fmt.Sprintf("cut/%d/%s",
			rule.UID, kodoSrc.Buckets[0].Bucket))
		vsRaw, _ := json.Marshal(vsaver)

		cutIntervalSecs := float64(0)
		_ = json.Unmarshal(rule.Automatic.Video.Params["cut_interval"],
			&cutIntervalSecs)

		vpreq := struct {
			Scenes []biz.Scene `json:"scenes,omitempty"`
			Params struct {
				Scenes  map[biz.Scene]json.RawMessage `json:"scenes"`
				Vframe  *vframe.VframeParams          `json:"vframe"`
				Save    json.RawMessage               `json:"save,omitempty"`
				HookURL string                        `json:"hookURL"`
			} `json:"params,omitempty"`
		}{
			Scenes: scenes,
			Params: struct {
				Scenes  map[biz.Scene]json.RawMessage `json:"scenes"`
				Vframe  *vframe.VframeParams          `json:"vframe"`
				Save    json.RawMessage               `json:"save,omitempty"`
				HookURL string                        `json:"hookURL"`
			}{
				Save: vsRaw,
			},
		}

		if cutIntervalSecs > 0 {
			mode := 0
			vpreq.Params.Vframe = &vframe.VframeParams{
				Mode:     &mode,           // 0表示自定义间隔，1表示关键帧
				Interval: cutIntervalSecs, // 单位秒
			}
		}

		vidParamsRaw, _ := json.Marshal(vpreq)
		req.Params.Video = vidParamsRaw
	}

	if kodoSrc.Buckets[0].Prefix != nil {
		req.Prefix = *kodoSrc.Buckets[0].Prefix
	}

	return &req, nil
}

func (b bJobs) New(ctx context.Context, rule *proto.Rule) (string, error) {
	xl := xlog.FromContextSafe(ctx)
	client := ahttp.NewQiniuStubRPCClient(rule.UID, rule.Utype, time.Second*30) // TODO
	var ret = struct {
		Job string `json:"job_id"` // 注意这里要与gate返回的json tag一致！
	}{}

	req, err := b.genNewReq(ctx, rule)
	if err != nil {
		xl.Errorf("genNewReq err, %+v", err)
		return "", err
	}

	param := struct {
		Request interface{} `json:"request"`
		HookURL string      `json:"hookURL"`
	}{
		Request: *req,
		HookURL: b.GetInnerAutoBjobNotifyURL(ctx, rule.UID, rule.RuleID),
	}

	xl.Infof("Call Bjob, %s, %s", JsonStr(param), b.Host)
	err = client.CallWithJson(ctx, &ret, "POST", b.Host, param)
	xl.Infof("Bjob Ret, %s, %v", JsonStr(ret), err)
	return ret.Job, err
}

func (b bJobs) Cancel(ctx context.Context, uid uint32, jobID string) error {
	return nil // TODO
}
