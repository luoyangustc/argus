package video_proxy

import (
	"fmt"
	"net/http"
	"time"

	"github.com/qiniu/rpc.v3"
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
