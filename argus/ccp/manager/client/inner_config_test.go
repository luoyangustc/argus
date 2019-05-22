package client

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestInnerCfg(t *testing.T) {

	ctx := context.Background()
	saver := Saver{
		UID:    123456,
		Bucket: "test-bucket",
		Prefix: "test-prefix",
	}
	mngHost := "http://test.qiniu.com"
	innerCfg := NewInnerConfig(&saver, mngHost)

	retSv := innerCfg.GetInnerSaver(ctx, "")
	assert.Equal(t, retSv.UID, saver.UID)
	assert.Equal(t, retSv.Zone, saver.Zone)
	assert.Equal(t, retSv.Bucket, saver.Bucket)
	assert.Equal(t, retSv.Prefix, saver.Prefix)

	pfopurl := innerCfg.GetInnerAutoPfopNotifyURL(ctx, 654321, "rlid_test", MT_IMAGE)
	assert.Equal(t, pfopurl, mngHost+"/v1/msg/pfop/654321/rlid_test/image")

	bjoburl := innerCfg.GetInnerAutoBjobNotifyURL(ctx, 654321, "rlid_test")
	assert.Equal(t, bjoburl, mngHost+"/v1/msg/bjob/654321/rlid_test")

	revwurl := innerCfg.GetInnerReviewNotifyURL(ctx, 654321, "rlid_test")
	assert.Equal(t, revwurl, mngHost+"/v1/msg/review/654321/rlid_test")

	mansuurl := innerCfg.GetInnerManualStreamNotifyURL(ctx, 654321, "rlid_test")
	assert.Equal(t, mansuurl, mngHost+"/v1/msg/manual/stream/654321/rlid_test")

	manbuurl := innerCfg.GetInnerManualBatchNotifyURL(ctx, 654321, "rlid_test")
	assert.Equal(t, manbuurl, mngHost+"/v1/msg/manual/batch/654321/rlid_test")
}
