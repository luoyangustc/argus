package client

import (
	"context"
	"encoding/json"
	"fmt"
)

const (
	MT_IMAGE = "image"
	MT_VIDEO = "video"
)

type Saver struct {
	UID    uint32 `json:"uid,omitempty"`
	Zone   int    `json:"zone,omitempty"`
	Bucket string `json:"bucket"`
	Prefix string `json:"prefix,omitempty"`
}

type InnerConfig interface {
	GetInnerSaver(ctx context.Context, pfxTag string) *Saver
	GetInnerAutoPfopNotifyURL(ctx context.Context, uid uint32, ruleID string, mimeType string) string
	GetInnerAutoBjobNotifyURL(ctx context.Context, uid uint32, ruleID string) string
	GetInnerManualStreamNotifyURL(ctx context.Context, uid uint32, ruleID string) string
	GetInnerManualBatchNotifyURL(ctx context.Context, uid uint32, ruleID string) string
	GetInnerReviewNotifyURL(ctx context.Context, uid uint32, ruleID string) string
}

type _InnerConfig struct {
	InnerSaver *Saver
	MngHost    string
}

// mngHost 带`HTTP`
func NewInnerConfig(innerSaver *Saver, mngHost string) InnerConfig {
	return &_InnerConfig{
		InnerSaver: innerSaver,
		MngHost:    mngHost,
	}
}

func (inner *_InnerConfig) GetInnerSaver(ctx context.Context, pfxTag string) *Saver {
	saver := Saver{
		UID:    inner.InnerSaver.UID,
		Zone:   inner.InnerSaver.Zone,
		Bucket: inner.InnerSaver.Bucket,
		Prefix: inner.InnerSaver.Prefix,
	}
	if pfxTag != "" {
		saver.Prefix += "/"
		saver.Prefix += pfxTag
	}
	return &saver
}

func (inner *_InnerConfig) GetInnerAutoPfopNotifyURL(ctx context.Context,
	uid uint32, ruleID string, mimeType string) string {
	return fmt.Sprintf("%s/v1/msg/pfop/%d/%s/%s", inner.MngHost, uid, ruleID, mimeType)
}

func (inner *_InnerConfig) GetInnerAutoBjobNotifyURL(ctx context.Context,
	uid uint32, ruleID string) string {
	return fmt.Sprintf("%s/v1/msg/bjob/%d/%s", inner.MngHost, uid, ruleID)
}

func (inner *_InnerConfig) GetInnerManualStreamNotifyURL(ctx context.Context, uid uint32, ruleID string) string {
	return fmt.Sprintf("%s/v1/msg/manual/stream/%d/%s", inner.MngHost, uid, ruleID)
}

func (inner *_InnerConfig) GetInnerManualBatchNotifyURL(ctx context.Context, uid uint32, ruleID string) string {
	return fmt.Sprintf("%s/v1/msg/manual/batch/%d/%s", inner.MngHost, uid, ruleID)
}

func (inner *_InnerConfig) GetInnerReviewNotifyURL(ctx context.Context,
	uid uint32, ruleID string) string {
	return fmt.Sprintf("%s/v1/msg/review/%d/%s", inner.MngHost, uid, ruleID)
}

//================

func JsonStr(obj interface{}) string {
	raw, err := json.Marshal(obj)
	if err != nil {
		return ""
	}
	return string(raw)
}
