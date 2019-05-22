package feature_group

import (
	"context"
	"time"

	xlog "github.com/qiniu/xlog.v1"
	ahttp "qiniu.com/argus/argus/com/http"
)

//----------------------------------------------------------------------------//

type Searcher interface {
	Search(context.Context, SearchReq) ([]SearchResult, error)
}

type SearchAPI struct {
	Host    string
	Path    string
	Timeout time.Duration
}

func (api SearchAPI) Search(ctx context.Context, req SearchReq) (
	ret []SearchResult, err error) {
	var (
		xl     = xlog.FromContextSafe(ctx)
		client = ahttp.NewQiniuStubRPCClient(1, 0, time.Second*60)

		f = func(ctx context.Context) error {
			return client.CallWithJson(ctx, &ret, "POST", api.Host+api.Path, req)
		}
	)
	_ = xl

	err = ahttp.CallWithRetry(ctx, []int{530}, []func(context.Context) error{f, f})
	return
}
