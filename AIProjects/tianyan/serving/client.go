package serving

import (
	"context"
	"fmt"
	"net/http"
	"time"

	"github.com/qiniu/http/httputil.v1"
	"github.com/qiniu/rpc.v3"
	"github.com/qiniu/xlog.v1"
)

const (
	defaultUID   uint32 = 1
	defaultUType uint32 = 2
)

type qiniuStubTransport struct {
	uid   uint32
	utype uint32
	http.RoundTripper
}

func (t qiniuStubTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	req.Header.Set(
		"Authorization",
		fmt.Sprintf("QiniuStub uid=%d&ut=%d", t.uid, t.utype),
	)
	return t.RoundTripper.RoundTrip(req)
}

func NewQiniuStubRPCClient(uid, utype uint32, timeout time.Duration) *rpc.Client {
	return &rpc.Client{
		Client: &http.Client{
			Timeout: timeout,
			Transport: qiniuStubTransport{
				uid:          uid,
				utype:        utype,
				RoundTripper: http.DefaultTransport,
			},
		},
	}
}

func NewDefaultStubRPCClient(timeout time.Duration) *rpc.Client {
	return &rpc.Client{
		Client: &http.Client{
			Timeout: timeout,
			Transport: qiniuStubTransport{
				uid:          defaultUID,
				utype:        defaultUType,
				RoundTripper: http.DefaultTransport,
			},
		},
	}
}

//----------------------------------------------------------------------------//

func CallWithRetry(
	ctx context.Context, skipCodes []int,
	calls []func(context.Context) error,
) (err error) {
	ctx2 := xlog.NewContext(ctx, xlog.FromContextSafe(ctx).Spawn())
	for _, call := range calls {

		err = call(ctx2)
		if err == nil {
			return
		}

		var retry = false
		code, _ := httputil.DetectError(err)
		for _, _code := range skipCodes {
			if code == _code {
				retry = true
				break
			}
		}
		if !retry {
			return
		}
	}
	return
}

func callRetry(ctx context.Context, f func(context.Context) error) error {
	return CallWithRetry(ctx, []int{530}, []func(context.Context) error{f, f})
}
