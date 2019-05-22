package util

import (
	"context"
	"fmt"

	xlog "github.com/qiniu/xlog.v1"
)

// SpawnContext 解决xlog库的并发问题，如果原context被cancel，这个函数返回的context也会被cancel
func SpawnContext(ctx context.Context) context.Context {
	return xlog.NewContext(ctx, xlog.FromContextSafe(ctx).Spawn())
}

func SpawnContext2(ctx context.Context, id int) context.Context {
	xl, ok := xlog.FromContext(ctx)
	if !ok {
		return xlog.NewContext(ctx, xlog.NewDummy())
	}
	return xlog.NewContextWith(ctx, fmt.Sprintf("%s.%d", xl.ReqId(), id))
}

// SpawnContextOnlyReqID 和SpawnContext的区别，如果原context被cancel，这个函数返回的context不受影响，仅仅用于保留reqid
func SpawnContextOnlyReqID(ctx context.Context) context.Context {
	reqID := xlog.FromContextSafe(ctx).ReqId()
	xl := xlog.NewWith(reqID)
	return xlog.NewContext(context.Background(), xl)
}
