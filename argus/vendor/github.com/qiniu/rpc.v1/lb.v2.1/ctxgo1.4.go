// +build go1.4
// +build !go1.5

// [[只是兼容下, 并不是支持go1.4]]

package lb

import (
	"net/http"

	"github.com/qiniu/rpc.v1"
	"github.com/qiniu/xlog.v1"
)

func (p *simple) doCtxWithHostRet(req *Request) (host string, resp *http.Response, code int, err error) {
	return p.doWithHostRet(req)
}

func (p *Client) DoWithHostRet(
	l rpc.Logger, req *Request) (host string, resp *http.Response, err error) {
	xl := xlog.NewWith(l)
	req.ctx = xlog.NewContext(req.Context(), xl)
	return p.DoCtxWithHostRet(req)
}
