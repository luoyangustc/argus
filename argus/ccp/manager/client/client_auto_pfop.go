package client

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/url"
	"strings"
	"time"

	xlog "github.com/qiniu/xlog.v1"
	"qbox.us/qconf/qconfapi"
	"qiniu.com/argus/argus/com/auth"
	ahttp "qiniu.com/argus/argus/com/http"
	"qiniu.com/argus/ccp/manager/proto"
	"qiniu.com/argus/ccp/manager/proto/kodo"
	"qiniu.com/argus/video/vframe"
	httputil "qiniupkg.com/http/httputil.v2"
)

type PfopRules interface {
	Set(ctx context.Context, rule *proto.Rule) error
	Del(ctx context.Context, rule *proto.Rule) error
	FillPfopName(ctx context.Context, rule *proto.Rule) error
}

var _ PfopRules = pfopRules{}

type pfopRules struct {
	Qconf  *qconfapi.Config
	UCHost string
	InnerConfig
}

func NewPfopRules(innerCfg InnerConfig, qconf *qconfapi.Config, ucHost string) pfopRules {
	return pfopRules{InnerConfig: innerCfg, Qconf: qconf, UCHost: ucHost}
}

type _PfopRulesSetReq struct {
	Bucket              string   `json:"bucket"`
	Name                string   `json:"name"`
	Prefix              string   `json:"prefix,omitempty"`
	MimeTypes           []string `json:"mime_types"`
	PersistentOPS       string   `json:"persistent_ops"`
	PersistentNotifyURL string   `json:"persistent_notify_url,omitempty"`
	PersistentPipeline  string   `json:"persistent_pipeline,omitempty"`
}

func (r pfopRules) FillPfopName(ctx context.Context, rule *proto.Rule) error {
	xl := xlog.FromContextSafe(ctx)

	_, kodoAction, err := kodo.UnmarshalRule(ctx, rule)
	if err != nil {
		xl.Errorf("kodo.UnmarshalRule err, %+v", err)
		return err
	}

	if rule.Image.IsOn && kodoAction.PfopName == nil {
		name := fmt.Sprintf("%s_%s_%s", rule.RuleID, MT_IMAGE,
			time.Now().Format("20060102150405"))
		kodoAction.PfopName = &name
	}

	if rule.Video.IsOn && kodoAction.PfopNameVideo == nil {
		name := fmt.Sprintf("%s_%s_%s", rule.RuleID, MT_VIDEO,
			time.Now().Format("20060102150405"))
		kodoAction.PfopNameVideo = &name
	}

	rawAct, err := json.Marshal(kodoAction)
	if err != nil {
		xl.Errorf("Marshal KodoAction err, %+v", err)
		return err
	}

	rule.Action = rawAct
	return nil
}

func (r pfopRules) genSetReq(ctx context.Context, rule *proto.Rule) (
	[]_PfopRulesSetReq, error) {

	reqArr := []_PfopRulesSetReq{}
	if rule.Image.IsOn {
		imgReq := _PfopRulesSetReq{}
		err := r.fillReqByMimeType(ctx, &imgReq, rule, MT_IMAGE)
		if err != nil {
			return nil, err
		}
		reqArr = append(reqArr, imgReq)
	}

	if rule.Video.IsOn {
		vidReq := _PfopRulesSetReq{}
		err := r.fillReqByMimeType(ctx, &vidReq, rule, MT_VIDEO)
		if err != nil {
			return nil, err
		}
		reqArr = append(reqArr, vidReq)
	}

	return reqArr, nil
}

func (r pfopRules) fillReqByMimeType(ctx context.Context, req *_PfopRulesSetReq,
	rule *proto.Rule, mimeType string) error {
	xl := xlog.FromContextSafe(ctx)

	kodoSrc, kodoAction, err := kodo.UnmarshalRule(ctx, rule)
	if err != nil {
		xl.Errorf("kodo.UnmarshalRule err, %+v", err)
		return err
	}

	if len(kodoSrc.Buckets) <= 0 {
		err := fmt.Errorf("kodoSrc.Buckets empty, %+v", kodoSrc)
		xl.Errorf("%+v", err)
		return err
	}

	prefix0 := ""
	if kodoSrc.Buckets[0].Prefix != nil {
		prefix0 = *kodoSrc.Buckets[0].Prefix
	}

	req.Bucket = kodoSrc.Buckets[0].Bucket
	req.Prefix = prefix0

	if mimeType == MT_IMAGE {
		req.MimeTypes = []string{"image/jpg", "image/jpeg", "image/png",
			"image/bmp", "image/gif"}
		req.PersistentOPS = "image-censor/v2/" + strings.Join(rule.Image.Scenes, "/")

		if kodoAction.PfopName == nil {
			err := fmt.Errorf("PfopName nil, %s", mimeType)
			xl.Errorf("%+v", err)
			return err
		}
		req.Name = *kodoAction.PfopName

	} else if mimeType == MT_VIDEO {
		req.MimeTypes = []string{"video/mp4", "video/x-m4v", "video/x-matroska",
			"video/webm", "video/quicktime", "video/x-msvideo",
			"video/x-ms-wmv", "video/mpeg", "video/x-flv",
			"video/x-ogm+ogg", "video/x-theora+ogg", "video/3gpp",
			"video/3gpp2", "video/annodex", "video/dv",
			"video/mp2t", "video/ogg", "video/vnd.mpegurl",
			"video/x-flic", "video/x-ms-asf", "video/x-nsv",
			"video/x-sgi-movie", "video/x-javafx"} // TODO
		req.PersistentOPS = "video-censor/v1/" + strings.Join(rule.Video.Scenes, "/")

		if rule.Saver.IsOn {
			vsaver := struct {
				Bucket string `json:"bucket"`
				Prefix string `json:"prefix,omitempty"`
			}{
				Bucket: rule.Saver.Bucket,
			}

			if rule.Saver.Prefix != nil {
				vsaver.Prefix = *rule.Saver.Prefix
			}

			vsRaw, _ := json.Marshal(vsaver)
			vsaveStr := base64.StdEncoding.EncodeToString(vsRaw)
			req.PersistentOPS += "/save/"
			req.PersistentOPS += vsaveStr
		}

		cutIntervalSecs := float64(0)
		_ = json.Unmarshal(rule.Automatic.Video.Params["cut_interval"],
			&cutIntervalSecs)
		if cutIntervalSecs > 0 { // 截帧间隔大于0，则传输帧参数
			mode := 0
			vframe := vframe.VframeParams{
				Mode:     &mode, // 0表示自定义间隔，1表示关键帧
				Interval: cutIntervalSecs,
			}

			vfRaw, _ := json.Marshal(vframe)
			vframeStr := base64.StdEncoding.EncodeToString(vfRaw)
			req.PersistentOPS += "/vframe/"
			req.PersistentOPS += vframeStr
		}

		if kodoAction.PfopNameVideo == nil {
			err := fmt.Errorf("PfopName nil, %s", mimeType)
			xl.Errorf("%+v", err)
			return err
		}
		req.Name = *kodoAction.PfopNameVideo

	} else {
		err := fmt.Errorf("invalid mimeType, %s", mimeType)
		xl.Errorf("%+v", err)
		return err
	}

	if kodoAction.Disable {
		xl.Infof("rule.Action: %+v", kodoAction)
		req.PersistentOPS += "|bucket-inspect/v2/true"
	} else {
		req.PersistentOPS += "|bucket-inspect/v2/false"
		xl.Infof("nil disable. %+v", req)
	}

	// notify-filter 设置内部回调地址
	req.PersistentOPS += "|notify-filter"
	if rule.Manual.IsOn {
		req.PersistentOPS += "/false"
	} else {
		req.PersistentOPS += "/true"
	}
	innerNotifyURL := r.GetInnerAutoPfopNotifyURL(ctx, rule.UID, rule.RuleID, mimeType)
	req.PersistentOPS += "/"
	req.PersistentOPS += url.QueryEscape(innerNotifyURL)

	if kodoAction.Pipeline != nil {
		req.PersistentPipeline = *kodoAction.Pipeline
	}

	if rule.NotifyURL != nil {
		// 正常设置用户自己的回调地址
		req.PersistentNotifyURL = *rule.NotifyURL
	}

	return nil
}

func (r pfopRules) Set(ctx context.Context, rule *proto.Rule) error {
	// return nil
	xl := xlog.FromContextSafe(ctx)

	ak, sk, err := auth.AkSk(qconfapi.New(r.Qconf), rule.UID)
	if err != nil {
		xl.Errorf("get aksk failed, %d, %+v", rule.UID, err)
		return err
	}

	client := ahttp.NewQboxAuthRPCClient(ak, sk, time.Second*30)
	reqArr, err := r.genSetReq(ctx, rule)
	if err != nil {
		xl.Errorf("pfop.genSetReq err, %+v", err)
		return err
	}

	for _, req := range reqArr {
		err := client.CallWithJson(ctx, nil, "POST",
			fmt.Sprintf("%s/pfopRules/set", r.UCHost), req)

		xl.Infof("pfopRulesSet, %s, %+v", JsonStr(req), err)
		if err != nil {
			return err
		}
	}
	return nil
}

func (r pfopRules) Del(ctx context.Context, rule *proto.Rule) error {
	xl := xlog.FromContextSafe(ctx)

	kodoSrc, kodoAction, err := kodo.UnmarshalRule(ctx, rule)
	if err != nil {
		xl.Errorf("kodo.UnmarshalRule err, %+v", err)
		return err
	}

	if len(kodoSrc.Buckets) <= 0 {
		err := fmt.Errorf("kodoSrc.Buckets empty, %+v", kodoSrc)
		xl.Errorf("%+v", err)
		return err
	}

	if rule.Image.IsOn {
		var name string
		if kodoAction.PfopName != nil {
			name = *kodoAction.PfopName
		} else { // 兼容老规则的删除
			name = fmt.Sprintf("image-censor-%s", strings.Join(rule.Image.Scenes, "_"))
		}
		err := r.delRule(ctx, rule.UID, kodoSrc.Buckets[0].Bucket, name)
		if err != nil {
			return err
		}
	}

	if rule.Video.IsOn {
		if kodoAction.PfopNameVideo == nil {
			err := fmt.Errorf("PfopName nil, %s", MT_VIDEO)
			xl.Errorf("%+v", err)
			return err
		}
		err := r.delRule(ctx, rule.UID, kodoSrc.Buckets[0].Bucket,
			*kodoAction.PfopNameVideo)
		if err != nil {
			return err
		}
	}

	return nil
}

func (r pfopRules) delRule(ctx context.Context, uid uint32, bucket, name string) error {
	xl := xlog.FromContextSafe(ctx)

	req := struct {
		Bucket string `json:"bucket"`
		Name   string `json:"name"`
	}{
		Bucket: bucket,
		Name:   name,
	}

	ak, sk, err := auth.AkSk(qconfapi.New(r.Qconf), uid)
	if err != nil {
		xl.Errorf("get aksk failed, %d, %+v", uid, err)
		return err
	}

	client := ahttp.NewQboxAuthRPCClient(ak, sk, time.Second*30)
	err = client.CallWithJson(ctx, nil, "POST",
		fmt.Sprintf("%s/pfopRules/delete", r.UCHost),
		req,
	)

	xl.Infof("pfopRulesDelete, %s, %+v", JsonStr(req), err)
	if err != nil {
		// print err & code
		// TIPS: 忽略该错误，幂等返回删除成功
		if err.Error() != "pfop rule not found" {
			ec := httputil.DetectCode(err)
			xl.Errorf("pfop rule delete err, %d, %+v", ec, err)
			return err
		}
	}

	return nil
}

//====
var _ PfopRules = MockPfopRules{}

type MockPfopRules struct {
}

func (mpr MockPfopRules) Set(ctx context.Context, rule *proto.Rule) error {
	return nil
}

func (mpr MockPfopRules) Del(ctx context.Context, rule *proto.Rule) error {
	return nil
}

func (mpr MockPfopRules) FillPfopName(ctx context.Context, rule *proto.Rule) error {
	return nil
}
