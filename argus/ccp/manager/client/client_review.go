package client

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	xlog "github.com/qiniu/xlog.v1"
	"gopkg.in/mgo.v2/bson"
	ahttp "qiniu.com/argus/argus/com/http"
	"qiniu.com/argus/ccp/manager/proto"
	"qiniu.com/argus/ccp/manager/proto/kodo"
)

// call review
type ReviewClient interface {
	NewSet(ctx context.Context, rule *proto.Rule) error
	PushItem(ctx context.Context, rule *proto.Rule, entry *Entry) error
	PushItems(ctx context.Context, rule *proto.Rule, batchEntries *BatchEntries) error
}

type _ReviewClient struct {
	InnerConfig
	Host string
}

func NewReviewClient(innerCfg InnerConfig, host string) ReviewClient {
	return &_ReviewClient{
		InnerConfig: innerCfg,
		Host:        host,
	}
}

type _NewSetReq struct {
	SetID      string `json:"set_id"`
	SourceType string `json:"source_type"`
	Type       string `json:"type"`
	Automatic  bool   `json:"automatic"`
	Manual     bool   `json:"manual"`
	Bucket     string `json:"bucket"`
	Prefix     string `json:"prefix"`
	NotifyURL  string `json:"notify_url"`
}

type BatchEntries struct {
	UID    uint32   `json:"uid"`
	Bucket string   `json:"bucket"`
	Keys   []string `json:"keys"`
}

type Entry struct {
	ID       bson.ObjectId   `json:"id"`
	SetID    string          `json:"set_id"` // For different sets
	Resource json.RawMessage `json:"resource,omitempty"`
	URIGet   string          `json:"uri_get,omitempty"`
	MimeType string          `json:"mimetype"`
	Version  string          `json:"version"`

	Original *OriginalSuggestion `json:"original"`
	Final    *FininalSuggestion  `json:"final"`
	// IMAGE: 推理结果
	// VIDEO: 各帧推理结果
	Result    json.RawMessage `json:"result,omitempty"`
	VideoCuts []VideoCut      `json:"video_cuts"`

	CreatedAt int64 `json:"created_at"`
}

type VideoCut struct {
	Uri      string              `bson:"uri" json:"uri"`
	Offset   int64               `bson:"offset" json:"offset"`
	Original *OriginalSuggestion `bson:"original" json:"original"`
}

type OriginalSuggestion struct {
	Source     string                               `json:"source"`
	Suggestion string                               `json:"suggestion"`
	Scenes     map[string]*OriginalSuggestionResult `json:"scenes"`
}

type OriginalSuggestionResult struct {
	Suggestion string      `json:"suggestion"`
	Labels     []LabelInfo `json:"labels"`
}

// 标签信息
type LabelInfo struct {
	Label string   `json:"label"`
	Score float32  `json:"score"`
	Group string   `json:"group,omitempty"`
	Pts   [][2]int `json:"pts"`
}

type FininalSuggestion struct {
	Suggestion string            `json:"suggestion"`
	Scenes     map[string]string `json:"scenes"`
}

func (r *_ReviewClient) NewSet(ctx context.Context, rule *proto.Rule) error {
	xl := xlog.FromContextSafe(ctx)

	kodoSrc, _, err := kodo.UnmarshalRule(ctx, rule)
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

	req := _NewSetReq{
		SetID:      rule.RuleID,
		SourceType: rule.SourceType,
		Type:       rule.Type,
		Automatic:  rule.Automatic.IsOn,
		Manual:     rule.Manual.IsOn,
		Bucket:     kodoSrc.Buckets[0].Bucket,
		Prefix:     prefix0,
		NotifyURL:  r.GetInnerReviewNotifyURL(ctx, rule.UID, rule.RuleID),
	}

	client := ahttp.NewQiniuStubRPCClient(rule.UID, rule.Utype, time.Second*30) // TODO
	url := fmt.Sprintf("%s/v1/sets", r.Host)
	xl.Infof("Call ReviewNewSet, %s, %s", JsonStr(req), url)
	err = client.CallWithJson(ctx, nil, "POST", url, req)
	xl.Infof("ReviewNewSet Ret, %+v", err)
	return err
}

func (r *_ReviewClient) PushItem(ctx context.Context, rule *proto.Rule, entry *Entry) error {
	xl := xlog.FromContextSafe(ctx)

	client := ahttp.NewQiniuStubRPCClient(rule.UID, rule.Utype, time.Second*30) // TODO
	url := fmt.Sprintf("%s/v1/sets/%s/entry", r.Host, entry.SetID)
	xl.Infof("Call ReviewAddEntry, %s, %s", JsonStr(entry), url)
	err := client.CallWithJson(ctx, nil, "POST", url, entry)
	xl.Infof("ReviewAddEntry Ret, %+v", err)
	return err
}

func (r *_ReviewClient) PushItems(ctx context.Context, rule *proto.Rule, batchEntries *BatchEntries) error {
	xl := xlog.FromContextSafe(ctx)

	client := ahttp.NewQiniuStubRPCClient(rule.UID, rule.Utype, time.Second*30) // TODO
	url := fmt.Sprintf("%s/v1/sets/%s/entries", r.Host, rule.RuleID)
	xl.Infof("Call ReviewAddEntries, %s, %s", JsonStr(batchEntries), url)
	err := client.CallWithJson(ctx, nil, "POST", url, batchEntries)
	xl.Infof("ReviewAddEntries Ret, %+v", err)
	return err
}

//====
var _ ReviewClient = MockReviewClient{}

type MockReviewClient struct {
}

func (mrc MockReviewClient) NewSet(ctx context.Context, rule *proto.Rule) error {
	return nil
}

func (mrc MockReviewClient) PushItem(ctx context.Context, rule *proto.Rule, entry *Entry) error {
	return nil
}

func (mrc MockReviewClient) PushItems(ctx context.Context, rule *proto.Rule, batchEntries *BatchEntries) error {
	return nil
}
