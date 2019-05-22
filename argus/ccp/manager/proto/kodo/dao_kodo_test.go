package kodo

import (
	"context"
	"encoding/json"
	"testing"

	mgoutil "github.com/qiniu/db/mgoutil.v3"
	"github.com/stretchr/testify/assert"
	"gopkg.in/mgo.v2/bson"
	"qiniu.com/argus/ccp/manager/proto"
)

func TestDaoKodo(t *testing.T) {

	var (
		ctx   = context.Background()
		colls struct {
			KodoSrcColl mgoutil.Collection `coll:"kodosrcs"`
		}
	)

	{
		mgoConf := &mgoutil.Config{DB: "CCP_UT"}
		sess, err := mgoutil.Open(&colls, mgoConf)
		assert.NoError(t, err)
		sess.SetPoolLimit(100)
		defer sess.Close()

		_, _ = colls.KodoSrcColl.RemoveAll(bson.M{})
	}

	kodoSrcDAO := NewKodoSrcDAO(&colls.KodoSrcColl)

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

	err := kodoSrcDAO.Create(ctx, &rule)
	assert.NoError(t, err)

	ruleRet, err := kodoSrcDAO.QueryBySrcID(ctx, uid, srcID)
	assert.NoError(t, err)
	assert.Equal(t, ruleRet.UID, uid)
	assert.Equal(t, ruleRet.SourceID, srcID)
	assert.Equal(t, ruleRet.Type, proto.TYPE_BATCH)

	rulesRet, err := kodoSrcDAO.Query(ctx, uid, srcRaw, nil)
	assert.NoError(t, err)
	assert.Equal(t, len(rulesRet), 1)

	batchType := proto.TYPE_BATCH
	rulesRet2, err := kodoSrcDAO.Query(ctx, uid, srcRaw, &batchType)
	assert.NoError(t, err)
	assert.Equal(t, len(rulesRet2), 1)

	streamType := proto.TYPE_STREAM
	rulesRet3, err := kodoSrcDAO.Query(ctx, uid, srcRaw, &streamType)
	assert.NoError(t, err)
	assert.Equal(t, len(rulesRet3), 0)
}
