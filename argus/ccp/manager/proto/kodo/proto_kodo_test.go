package kodo

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"qiniu.com/argus/ccp/manager/proto"
)

func TestProtoKodo(t *testing.T) {

	uid := uint32(12345)
	bucket := "bucket001"
	prefix := "prefix001"
	ruleID := "rule001"
	srcID := "src001"

	kodoSrc := KodoSrc{
		Buckets: []struct {
			Bucket string  `json:"bucket"`
			Prefix *string `json:"prefix,omitempty"`
		}{
			struct {
				Bucket string  `json:"bucket"`
				Prefix *string `json:"prefix,omitempty"`
			}{
				Bucket: bucket,
				Prefix: &prefix,
			},
		},
	}

	srcRaw, _ := json.Marshal(kodoSrc)

	rule := proto.Rule{
		RuleID:     ruleID,
		UID:        uid,
		SourceType: proto.SRC_KODO,
		SourceID:   srcID,
		Source:     srcRaw,
		Status:     proto.RULE_STATUS_ON,
		Type:       proto.TYPE_BATCH,
	}
	kodoSrcInMgo := KodoSrcInMgo{}
	err := convertRuleToKodoSrc(&rule, &kodoSrcInMgo)
	assert.NoError(t, err)
	assert.Equal(t, kodoSrcInMgo.UID, uid)
	assert.Equal(t, kodoSrcInMgo.Type, proto.TYPE_BATCH)

	ruleRet := proto.Rule{}
	err = convertKodoSrcToRule(&kodoSrcInMgo, &ruleRet)
	assert.NoError(t, err)
	assert.Equal(t, ruleRet.UID, uid)
	assert.Equal(t, ruleRet.Type, proto.TYPE_BATCH)

}
