package util

import (
	"context"
	"fmt"
	"net/http"

	xlog "github.com/qiniu/xlog.v1"
)

func CtxAndLog(
	ctx context.Context, w http.ResponseWriter, req *http.Request,
) (context.Context, *xlog.Logger) {

	xl, ok := xlog.FromContext(ctx)
	if !ok {
		xl = xlog.New(w, req)
		ctx = xlog.NewContext(ctx, xl)
	}
	return ctx, xl
}

func SpawnContext(ctx context.Context) context.Context {
	return xlog.NewContext(ctx, xlog.FromContextSafe(ctx).Spawn())
}

////////////////////////////////////////////////////////////////////////////////

func SetStateHeader(header http.Header, name string, state int) {
	const key = "X-Origin-A"

	if _, ok := header[key]; ok {
		header.Set(key, header.Get(key)+";"+fmt.Sprintf("%s:%d", name, state))
	} else {
		header.Set(key, fmt.Sprintf("%s:%d", name, state))
	}
}
