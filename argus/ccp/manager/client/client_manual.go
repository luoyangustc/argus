package client

import (
	"context"
	"fmt"
	"time"

	xlog "github.com/qiniu/xlog.v1"
	ahttp "qiniu.com/argus/argus/com/http"
	"qiniu.com/argus/ccp/manager/proto"
)

// call manual
type ManualJobs interface {
	NewSet(ctx context.Context, rule *proto.Rule) error
	PushItem(ctx context.Context, rule *proto.Rule, entry interface{}) error
	PushItems(ctx context.Context, rule *proto.Rule, batchEntries *MJBatchEntries) error
}

func NewManualJobs(innerCfg InnerConfig, host string) ManualJobs {
	return &_MJobs{
		InnerConfig: innerCfg,
		Host:        host,
	}
}

type _MJobs struct {
	InnerConfig
	Host string
}

type MJBatchEntries struct {
	UID    uint32   `json:"uid"`
	Bucket string   `json:"bucket"`
	Keys   []string `json:"keys"`
}

func (mj *_MJobs) NewSet(ctx context.Context, rule *proto.Rule) error {
	xl := xlog.FromContextSafe(ctx)

	var ntfUrl string
	if rule.Type == proto.TYPE_STREAM {
		ntfUrl = mj.GetInnerManualStreamNotifyURL(ctx, rule.UID, rule.RuleID)
	} else {
		ntfUrl = mj.GetInnerManualBatchNotifyURL(ctx, rule.UID, rule.RuleID)
	}

	req := struct {
		SetId      string `json:"set_id,omitempty"`
		UID        uint32 `json:"uid"`
		SourceType string `json:"source_type,omitempty"` // KODO | API
		Type       string `json:"type,omitempty"`        // STREAM | BATCH
		Image      struct {
			IsOn   bool     `json:"is_on"`
			Scenes []string `json:"scenes,omitempty"` // pulp | terror | politician |...
		} `json:"image,omitempty"`
		Video struct {
			IsOn   bool     `json:"is_on"`
			Scenes []string `json:"scenes,omitempty"` // pulp | terror | politician |...
		} `json:"video,omitempty"`
		NotifyURL string `json:"notify_url,omitempty"`
	}{
		SetId:      rule.RuleID,
		UID:        rule.UID,
		SourceType: rule.SourceType,
		Type:       rule.Type,
		Image:      rule.Image,
		Video:      rule.Video,
		NotifyURL:  ntfUrl,
	}

	client := ahttp.NewQiniuStubRPCClient(rule.UID, rule.Utype, time.Second*30) // TODO
	url := fmt.Sprintf("%s/v1/ccp/manual/sets", mj.Host)
	xl.Infof("Call ManualNewSet, %s, %s", JsonStr(req), url)
	err := client.CallWithJson(ctx, nil, "POST", url, req)
	xl.Infof("ManualNewSet Ret, %+v", err)
	return err
}

func (mj *_MJobs) PushItem(ctx context.Context, rule *proto.Rule, entry interface{}) error {

	// TODO
	return nil
}

func (mj *_MJobs) PushItems(ctx context.Context, rule *proto.Rule, batchEntries *MJBatchEntries) error {
	xl := xlog.FromContextSafe(ctx)

	client := ahttp.NewQiniuStubRPCClient(rule.UID, rule.Utype, time.Second*30) // TODO
	url := fmt.Sprintf("%s/v1/ccp/manual/sets/%s/entries", mj.Host, rule.RuleID)
	xl.Infof("Call ManualAddEntries, %s, %s", JsonStr(batchEntries), url)
	err := client.CallWithJson(ctx, nil, "POST", url, batchEntries)
	xl.Infof("ManualAddEntries Ret, %+v", err)
	return err
}

//====

var _ ManualJobs = MockManualJobs{}

type MockManualJobs struct {
}

func (mmj MockManualJobs) NewSet(ctx context.Context, rule *proto.Rule) error {
	return nil
}

func (mmj MockManualJobs) PushItem(ctx context.Context, rule *proto.Rule, entry interface{}) error {
	return nil
}

func (mmj MockManualJobs) PushItems(ctx context.Context, rule *proto.Rule, batchEntries *MJBatchEntries) error {
	return nil
}
