package service

import (
	"context"
	"encoding/base64"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strings"

	qhttputil "github.com/qiniu/http/httputil.v1"
	"github.com/qiniu/http/restrpc.v1"
)

type URIProxy interface {
	URI(string) string
	GetUri_(context.Context,
		*struct {
			CmdArgs []string
		},
		*restrpc.Env,
	)
}

var _ URIProxy = (*uriProxy)(nil)

type uriProxy struct {
	prefix string
}

func NewURIProxy(prefix string) URIProxy {
	if !strings.HasPrefix(prefix, "/") {
		prefix = "/" + prefix
	}
	if !strings.HasSuffix(prefix, "/") {
		prefix = prefix + "/"
	}
	prefix = prefix + "uri/"
	return &uriProxy{prefix: prefix}
}

func (p *uriProxy) URI(uri string) string {
	return p.prefix + base64.URLEncoding.EncodeToString([]byte(uri))
}

func (p *uriProxy) GetUri_(ctx context.Context,
	req *struct {
		CmdArgs []string
	},
	env *restrpc.Env,
) {

	bs, err := base64.URLEncoding.DecodeString(req.CmdArgs[0])
	if err != nil {
		qhttputil.ReplyErr(env.W, http.StatusBadRequest, "bad uri")
		return
	}
	target, err := url.Parse(string(bs))
	if err != nil {
		qhttputil.ReplyErr(env.W, http.StatusBadRequest, "bad uri")
		return
	}

	director := func(req *http.Request) {
		req.URL = target
		req.Host = target.Host
		if _, ok := req.Header["User-Agent"]; !ok {
			// explicitly disable User-Agent so it's not set to default value
			req.Header.Set("User-Agent", "")
		}
	}

	proxy := &httputil.ReverseProxy{Director: director}
	proxy.ServeHTTP(env.W, env.Req)
}
