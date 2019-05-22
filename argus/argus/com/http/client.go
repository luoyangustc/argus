package http

import (
	"context"
	"fmt"
	"net/http"
	"time"

	"github.com/qiniu/http/httputil.v1"
	"github.com/qiniu/rpc.v3"
	"qiniu.com/auth/qboxmac.v1"
	"qiniu.com/auth/qiniumac.v1"
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

func NewQiniuAuthRPCClient(ak, sk string, timeout time.Duration) *rpc.Client {
	return &rpc.Client{
		Client: &http.Client{
			Timeout: timeout,
			Transport: qiniumac.NewTransport(
				&qiniumac.Mac{AccessKey: ak, SecretKey: []byte(sk)},
				http.DefaultTransport,
			),
		},
	}
}

func NewQboxAuthRPCClient(ak, sk string, timeout time.Duration) *rpc.Client {
	return &rpc.Client{
		Client: &http.Client{
			Timeout: timeout,
			Transport: qboxmac.NewTransport(
				&qboxmac.Mac{AccessKey: ak, SecretKey: []byte(sk)},
				http.DefaultTransport,
			),
		},
	}
}

//----------------------------------------------------------------------------//

func CallWithRetry(
	ctx context.Context, skipCodes []int,
	calls []func(context.Context) error,
) (err error) {
	for _, call := range calls {

		err = call(ctx)
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
