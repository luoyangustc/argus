package proxy

import (
	"context"
	"encoding/base64"
	"time"

	"github.com/qiniu/http/restrpc.v1"

	ahttp "qiniu.com/argus/argus/com/http"
	"qiniu.com/argus/com/proxy/fop"
)

var _ fop.Proxy = Proxy{}

type Proxy struct {
	URL string
}

func NewProxy(_url string) fop.Proxy { return Proxy{URL: _url} }

func (p Proxy) Post(ctx context.Context, req fop.ProxyReq, env *restrpc.Env) (interface{}, error) {

	var uri = req.URL

	if req.URL == "" {
		uri = "data:application/octet-stream;base64," +
			base64.StdEncoding.EncodeToString(req.ReqBody)
		// return nil, httputil.NewError(http.StatusInternalServerError, "unsupport post")
	}

	var (
		client = ahttp.NewQiniuStubRPCClient(uint32(req.UID), 4, time.Second*60)
		ret    = new(interface{})
		call   = func(ctx context.Context) error {
			return client.CallWithJson(ctx, ret, "POST", p.URL,
				struct {
					Data struct {
						URI string `json:"uri"`
					} `json:"data"`
				}{
					Data: struct {
						URI string `json:"uri"`
					}{
						URI: uri,
					},
				})
		}
	)

	err := ahttp.CallWithRetry(ctx, []int{530}, []func(context.Context) error{call, call})

	return ret, err
}
