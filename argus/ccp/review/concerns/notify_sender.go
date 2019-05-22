package concerns

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"time"

	xlog "github.com/qiniu/xlog.v1"
	"qiniu.com/argus/ccp/review/dao"
	"qiniu.com/argus/ccp/review/model"
)

var (
	NotifySender = NewNotifySender()
)

type _NotifySender struct {
	*http.Client
}

func NewNotifySender() *_NotifySender {
	return &_NotifySender{
		Client: &http.Client{
			Transport: &http.Transport{
				MaxIdleConns:       50,
				IdleConnTimeout:    30 * time.Second,
				DisableCompression: true,
			},
		},
	}
}

const (
	contentType = "application/json"
)

func (this *_NotifySender) Perform(ctx context.Context, entry *model.Entry) {
	var (
		success bool
		xl      = xlog.FromContextSafe(ctx)
	)

	defer func() {
		if success {
			xl.Errorf("entry alert notify failed: <%s>", entry.ID.Hex())
		} else {
			xl.Infof("entry alert notify success: <%s>", entry.ID.Hex())
		}
	}()

	set, err := dao.EntrySetCache.MustGet(entry.SetId)
	if err != nil {
		xl.Errorf("dao.EntrySetCache.MustGet: %v", err)
		return
	}

	if set.NotifyURL == "" {
		xl.Errorf("no notify url found: <%s>", set.SetId)
		return
	}

	alert := model.NewNotifyAlert(
		set.SourceType,
		entry.SetId,
		set.Bucket,
		entry.URIGet,
		entry.Original.Suggestion,
		entry.Final.Suggestion,
	)

	payload, err := json.Marshal(alert)
	if err != nil {
		xl.Errorf("json.Marshal: %v", err)
		return
	}
	resp, err := this.Post(set.NotifyURL, contentType, bytes.NewReader(payload))
	if err != nil {
		xl.Errorf("send alert with error: <%s>, %v", set.NotifyURL, err)
		return
	}
	defer resp.Body.Close()

	// make sure it's 2xx
	success = resp.StatusCode/100 == 2
	if !success {
		xl.Warnf("send alert with response code: <%s>, %d", set.NotifyURL, resp.StatusCode)
	}
}
