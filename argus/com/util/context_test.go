package util

import (
	"context"
	"testing"

	"github.com/stretchr/testify.v2/assert"

	xlog "github.com/qiniu/xlog.v1"
)

func TestSpawnContext(t *testing.T) {
	xl := xlog.NewWith("test")
	ctx2 := xlog.NewContext(context.Background(), xl)
	ctx3 := SpawnContextOnlyReqID(ctx2)
	ctx4 := SpawnContext(ctx2)
	xl3 := xlog.FromContextSafe(ctx3)
	xl4 := xlog.FromContextSafe(ctx4)
	assert.Equal(t, xl.ReqId(), xl3.ReqId())
	assert.Equal(t, xl.ReqId(), xl4.ReqId())
}

func TestSpawnContextCancel(t *testing.T) {
	xl := xlog.NewWith("test")
	ctx, cancel := context.WithCancel(context.Background())
	ctx2 := xlog.NewContext(ctx, xl)
	ctx3 := SpawnContextOnlyReqID(ctx2)
	ctx4 := SpawnContext(ctx2)
	xl3 := xlog.FromContextSafe(ctx3)
	xl4 := xlog.FromContextSafe(ctx4)
	assert.Equal(t, xl.ReqId(), xl3.ReqId())
	assert.Equal(t, xl.ReqId(), xl4.ReqId())

	cancel()
	assert.Equal(t, nil, ctx3.Err())
	assert.Equal(t, context.Canceled, ctx4.Err())
}
