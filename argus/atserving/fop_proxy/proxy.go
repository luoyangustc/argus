package fop_proxy

import (
	"context"
	"net/http"
	"strings"
	"time"

	"github.com/qiniu/http/httputil.v1"
	"github.com/qiniu/http/restrpc.v1"

	"qiniu.com/argus/com/proxy/fop"
)

var _ fop.Proxy = Proxy{}

type Proxy struct {
	Host string
}

func NewProxy(host string) fop.Proxy { return Proxy{Host: host} }

func (p Proxy) Post(ctx context.Context, req fop.ProxyReq, env *restrpc.Env) (interface{}, error) {

	items := strings.Split(req.Cmd, "/")
	if len(items) < 2 || items[0] != "eval" {
		return nil, httputil.NewError(http.StatusBadRequest, "bad cmd: "+req.Cmd)
	}

	var (
		cmd    = items[1]
		url    = p.Host + "/v1/eval/" + cmd
		client = fop.NewQiniuStubRPCClient(uint32(req.UID), 4, time.Second*60)
		ret    = new(interface{})
		err    error
	)

	if req.URL != "" {
		err = client.CallWithJson(ctx, ret, "POST", url,
			struct {
				Data struct {
					URI string `json:"uri"`
				} `json:"data"`
			}{
				Data: struct {
					URI string `json:"uri"`
				}{
					URI: req.URL,
				},
			})
	} else {
		err = client.CallWith64(ctx, ret,
			"POST", url,
			"application/octet-stream",
			env.Req.Body, env.Req.ContentLength,
		)
	}

	return ret, err
}
