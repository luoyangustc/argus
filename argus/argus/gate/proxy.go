package gate

import (
	"context"
	"crypto/tls"
	"fmt"
	"net/http"
	"net/http/httputil"
	"strings"
	"sync"
	"time"

	qhttputil "github.com/qiniu/http/httputil.v1"
	"qiniu.com/auth/authstub.v1"
	"qiniu.com/argus/argus/com/util"
)

type ProxyRoute struct {
	Path    string  `json:"path"`
	Host    string  `json:"host"`
	Replace string  `json:"replace"`
	TLS     TlsPair `json:"tls"`

	Public        bool     `json:"public"`
	UserWhiteList []uint32 `json:"user_whitelist"`
}

type TlsPair struct {
	CertPem string `json:"cert_pem"`
	CertKey string `json:"cert_key"`
}

type ProxyRoutes struct {
	rs []ProxyRoute
	*sync.RWMutex
}

func NewProxyRoutes() *ProxyRoutes {
	return &ProxyRoutes{
		rs:      make([]ProxyRoute, 0),
		RWMutex: new(sync.RWMutex),
	}
}

func (r *ProxyRoutes) Set(path string, route ProxyRoute) {
	r.Lock()
	defer r.Unlock()
	var (
		rs = make([]ProxyRoute, 0, len(r.rs)+1)
		j  = -1
	)
	for i, _route := range r.rs {
		if j < 0 {
			if len(_route.Path) < len(route.Path) {
				rs = append(rs, route)
				j = i
			} else if _route.Path == route.Path {
				rs = append(rs, route)
				j = i
				continue
			}
		}
		rs = append(rs, _route)
	}
	if j < 0 {
		rs = append(rs, route)
	}
	r.rs = rs
}

func (r *ProxyRoutes) Del(path string) {
	r.Lock()
	defer r.Unlock()
	var (
		rs    = make([]ProxyRoute, len(r.rs)-1)
		index = -1
	)
	for i, route := range r.rs {
		if route.Path == path {
			index = i
			break
		}
	}
	if index < 0 {
		return
	}
	copy(rs, r.rs[:index])
	copy(rs[index:], r.rs[index+1:])
	r.rs = rs
}

func (r *ProxyRoutes) Get(path string) (ProxyRoute, bool) {

	match := func(pattern, path string) bool {
		if len(pattern) == 0 {
			// should not happen
			return false
		}
		n := len(pattern)
		if pattern[n-1] != '/' {
			return pattern == path
		}
		return len(path) >= n && path[0:n] == pattern
	}

	r.RLock()
	defer r.RUnlock()

	var (
		index = -1
		n     = 0
	)
	for i, v := range r.rs {
		if !match(v.Path, path) {
			continue
		}
		if index < 0 || len(v.Path) > n {
			n = len(v.Path)
			index = i
		}
	}
	if index < 0 {
		return ProxyRoute{}, false
	}
	return r.rs[index], true
}

type Proxy struct {
	Routes *ProxyRoutes
}

func (p *Proxy) Do(ctx context.Context, env *authstub.Env) {

	ctx, xl := util.CtxAndLog(ctx, env.W, env.Req)

	xl.Infof("PROXY Req: %#v", env.Req)

	route, ok := p.Routes.Get(env.Req.URL.Path)
	if !ok {
		xl.Warnf("Proxy 404: %v", env.Req.URL.Path)
		qhttputil.ReplyErr(env.W, http.StatusNotFound, "Page Not Found")
		return
	}

	if !route.Public {
		found := false
		for _, uid := range route.UserWhiteList {
			if uid == env.Uid {
				found = true
				break
			}
		}
		if !found {
			xl.Warnf("Proxy 401: %d %#v", env.Uid, route)
			qhttputil.ReplyErr(env.W, http.StatusNotFound, "Page Not Found")
			return
		}
	}

	var (
		_api = strings.Replace(route.Path, "/", "_", -1)
	)
	requestsParallelAtProxy(_api).Inc()
	defer func(begin time.Time) {
		requestsParallelAtProxy(_api).Dec()
		responseTimeAtProxy(_api, "").Observe(durationAsFloat64(time.Since(begin)))
	}(time.Now())

	proxy := &httputil.ReverseProxy{
		Director: func(req *http.Request) {

			req.URL.Scheme = "http"
			req.URL.Host = route.Host

			if len(route.Replace) > 0 {
				req.URL.Path = strings.Replace(req.URL.Path, route.Path, route.Replace, 1)
			}

			if strings.TrimSpace(route.TLS.CertKey) != "" && strings.TrimSpace(route.TLS.CertPem) != "" {
				req.URL.Scheme = "https"
			}

			req.Host = route.Host
			req.RequestURI = req.URL.Path

			if _, ok := req.Header["User-Agent"]; !ok {
				// explicitly disable User-Agent so it's not set to default value
				req.Header.Set("User-Agent", "")
			}
			req.Header.Set(
				"Authorization",
				fmt.Sprintf("QiniuStub uid=%d&ut=%d", env.Uid, env.Utype),
			)

			xl.Infof("PROXY: %v", req.URL.Path)
		},
	}

	if strings.TrimSpace(route.TLS.CertKey) != "" && strings.TrimSpace(route.TLS.CertPem) != "" {
		cert, err := tls.X509KeyPair([]byte(route.TLS.CertPem), []byte(route.TLS.CertKey))
		if err != nil {
			xl.Warnf("load tls key pair error: %d %#v", env.Uid, route)
			qhttputil.ReplyErr(env.W, http.StatusInternalServerError, "Load tls key pair error")
			return
		}

		proxy.Transport = &http.Transport{
			TLSClientConfig: &tls.Config{
				Certificates:       []tls.Certificate{cert},
				InsecureSkipVerify: true,
			},
		}
	}

	proxy.ServeHTTP(env.W, env.Req)
}
