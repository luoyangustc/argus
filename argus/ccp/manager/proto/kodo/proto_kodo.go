package kodo

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	xlog "github.com/qiniu/xlog.v1"
	"qiniu.com/argus/ccp/manager/proto"
)

//================================================================

type KodoSrc struct {
	Buckets []struct {
		Bucket string  `json:"bucket"`
		Prefix *string `json:"prefix,omitempty"`
	} `json:"buckets"`
}

type KodoAction struct {
	Disable       bool                       `json:"disable"`
	Threshold     map[string]json.RawMessage `json:"threshold"`
	Pipeline      *string                    `json:"pipeline,omitempty"`
	PfopName      *string                    `json:"pfop_name,omitempty"`       // For Image
	PfopNameVideo *string                    `json:"pfop_name_video,omitempty"` // For Video
}

//================================================================

type KodoSrcInMgo struct {
	SourceID   string  `json:"source_id" bson:"source_id"`
	SourceType string  `json:"source_type" bson:"source_type"` // KODO | API
	UID        uint32  `json:"uid" bson:"uid"`
	Utype      uint32  `json:"utype,omitempty" bson:"utype,omitempty"`
	Bucket     string  `json:"bucket" bson:"bucket"`
	Prefix     *string `json:"prefix,omitempty" bson:"prefix,omitempty"`
	Type       string  `json:"type" bson:"type"` // STREAM | BATCH

	CreateTime time.Time `json:"create_time" bson:"create_time"`
}

//================================================================
// Converter

func convertRuleToKodoSrc(rule *proto.Rule, kodoSrcInMgo *KodoSrcInMgo) error {

	if rule == nil || kodoSrcInMgo == nil {
		return errors.New("convert failed, invalid params")
	}

	kodoSrcInMgo.SourceType = rule.SourceType
	kodoSrcInMgo.UID = rule.UID
	kodoSrcInMgo.Utype = rule.Utype
	kodoSrcInMgo.Type = rule.Type
	kodoSrcInMgo.SourceID = rule.SourceID
	kodoSrcInMgo.CreateTime = time.Unix(rule.CreateSec, 0)

	kodoSrc := KodoSrc{}
	err := json.Unmarshal(rule.Source, &kodoSrc)
	if err != nil {
		return err
	}

	if len(kodoSrc.Buckets) <= 0 {
		return errors.New("convert failed, KodoSrc.Buckets empty")
	}

	// 暂只取一个Bucket来源
	kodoSrcInMgo.Bucket = kodoSrc.Buckets[0].Bucket
	kodoSrcInMgo.Prefix = kodoSrc.Buckets[0].Prefix

	return nil
}

func convertKodoSrcToRule(kodoSrcInMgo *KodoSrcInMgo, rule *proto.Rule) error {

	if rule == nil || kodoSrcInMgo == nil {
		return errors.New("convert failed, invalid params")
	}

	kodoSrc := KodoSrc{
		Buckets: []struct {
			Bucket string  `json:"bucket"`
			Prefix *string `json:"prefix,omitempty"`
		}{
			struct {
				Bucket string  `json:"bucket"`
				Prefix *string `json:"prefix,omitempty"`
			}{
				Bucket: kodoSrcInMgo.Bucket,
				Prefix: kodoSrcInMgo.Prefix,
			},
		},
	}

	srcRaw, err := json.Marshal(kodoSrc)
	if err != nil {
		return err
	}

	rule.SourceType = kodoSrcInMgo.SourceType
	rule.Source = srcRaw

	rule.UID = kodoSrcInMgo.UID
	rule.Utype = kodoSrcInMgo.Utype
	rule.Type = kodoSrcInMgo.Type
	rule.SourceID = kodoSrcInMgo.SourceID
	rule.CreateSec = kodoSrcInMgo.CreateTime.Unix()

	return nil
}

// 解析rule
func UnmarshalRule(ctx context.Context, rule *proto.Rule) (
	*KodoSrc, *KodoAction, error) {
	xl := xlog.FromContextSafe(ctx)

	kodoSrc := KodoSrc{}
	err := json.Unmarshal(rule.Source, &kodoSrc)
	if err != nil {
		xl.Errorf("UnmarshalRule err, %+v", err)
		return nil, nil, err
	}
	if len(kodoSrc.Buckets) <= 0 {
		err = fmt.Errorf("KodoSrc.Buckets empty, %+v", kodoSrc)
		xl.Errorf("UnmarshalRule err, %+v", err)
		return nil, nil, err
	}

	kodoAction := KodoAction{}
	if len(rule.Action) > 0 {
		err := json.Unmarshal(rule.Action, &kodoAction)
		if err != nil {
			xl.Errorf("UnmarshalRule err, %+v", err)
			return nil, nil, err
		}
	}

	return &kodoSrc, &kodoAction, nil
}
