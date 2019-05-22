package vframe

import (
	"context"
	"encoding/base64"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strings"

	qhttputil "github.com/qiniu/http/httputil.v1"
	"github.com/qiniu/http/restrpc.v1"

	STS "qiniu.com/argus/sts/client"
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

var _ URIProxy = uriProxy{}

type uriProxy struct {
	prefix string
}

func NewURIProxy(prefix string) uriProxy {
	return uriProxy{prefix: prefix}
}

func (p uriProxy) URI(uri string) string {
	if p.prefix[len(p.prefix)-1] != '/' {
		return p.prefix + "/" + base64.URLEncoding.EncodeToString([]byte(uri))
	}

	return p.prefix + base64.URLEncoding.EncodeToString([]byte(uri))
}

func (p uriProxy) GetUri_(ctx context.Context,
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
	target, _ := url.Parse(string(bs))
	director := func(req *http.Request) {
		req.URL = target
		req.Host = target.Host
		if _, ok := req.Header["User-Agent"]; !ok {
			// explicitly disable User-Agent so it's not set to default value
			req.Header.Set("User-Agent", "")
		}
		if _, ok := req.Header["Authorization"]; ok {
			req.Header.Del("Authorization")
		}
	}
	proxy := &httputil.ReverseProxy{Director: director}
	proxy.ServeHTTP(env.W, env.Req)

}

////////////////////////////////////////////////////////////////////////////////

var _ URIProxy = stsProxy{}

type stsProxy struct {
	prefix string
	client STS.Client
}

func NewSTSProxy(prefix string, cli STS.Client) stsProxy {
	return stsProxy{prefix: prefix, client: cli}
}

func (p stsProxy) URI(uri string) string {
	if p.prefix[len(p.prefix)-1] != '/' {
		return p.prefix + "/" + base64.URLEncoding.EncodeToString([]byte(uri))
	}

	return p.prefix + base64.URLEncoding.EncodeToString([]byte(uri))
}

func (p stsProxy) GetUri_(ctx context.Context,
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
	director := func(req *http.Request) {
		var options = STS.OPTION_PROXY
		uri, _ := p.client.GetURL(ctx, string(bs), nil, &options)
		uri = strings.Replace(uri, STS.SCHEME_STS, STS.SCHEME_HTTP, 1)
		target, _ := url.Parse(uri)
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
