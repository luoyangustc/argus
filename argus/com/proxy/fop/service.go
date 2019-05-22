package fop

import (
	"context"
	"encoding/base64"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strconv"

	qhttputil "github.com/qiniu/http/httputil.v1"
	"github.com/qiniu/http/restrpc.v1"
)

type ProxyReq struct {
	Cmd     string `json:"cmd"`
	UID     uint32 `json:"uid"`
	URL     string `json:"url"`
	ReqBody []byte `json:"req_body"`
}

type Proxy interface {
	Post(context.Context, ProxyReq, *restrpc.Env) (interface{}, error)
}

//----------------------------------------------------------------------------//

type _DataReq struct {
	URL string `json:"url"`
}

type _HandlerReq struct {
	Cmd string `json:"cmd"`
	URL string `json:"url"`
}

type Service interface {
	GetData(context.Context, *_DataReq, *restrpc.Env)
	PostHandler(context.Context, *_HandlerReq, *restrpc.Env) (interface{}, error)
}

type service struct {
	Host string
	Proxy
}

func NewService(host string, proxy Proxy) Service {
	return service{Host: host, Proxy: proxy}
}

func (s service) GetData(ctx context.Context, req *_DataReq, env *restrpc.Env) {

	target, _ := url.Parse(req.URL)
	director := func(req *http.Request) {
		req.URL = target
		if _, ok := req.Header["User-Agent"]; !ok {
			// explicitly disable User-Agent so it's not set to default value
			req.Header.Set("User-Agent", "")
		}
	}
	proxy := &httputil.ReverseProxy{Director: director}
	proxy.ServeHTTP(env.W, env.Req)

}

func (s service) PostHandler(
	ctx context.Context, req *_HandlerReq, env *restrpc.Env,
) (interface{}, error) {

	if req.Cmd == "" {
		vs, err := url.ParseQuery(env.Req.URL.RawQuery)
		if err != nil {
			return nil, qhttputil.NewError(http.StatusBadRequest, err.Error())
		}
		req.Cmd = vs.Get("cmd")
		req.URL = vs.Get("url")
	}

	if req.Cmd == "" {
		return nil, qhttputil.NewError(http.StatusBadRequest, "need params: cmd")
	}

	var uid uint64
	{
		str := env.Req.Header.Get("X-Qiniu-Uid")
		if str == "" {
			return nil, qhttputil.NewError(http.StatusBadRequest, "need params: uid")
		}
		bs, err := base64.StdEncoding.DecodeString(str)
		if err != nil {
			return nil, qhttputil.NewError(
				http.StatusBadRequest,
				fmt.Sprintf("bad params: uid, %s", str),
			)
		}
		uid, err = strconv.ParseUint(string(bs), 10, 64)
		if err != nil {
			return nil, qhttputil.NewError(
				http.StatusBadRequest,
				fmt.Sprintf("bad params: uid, %s", string(bs)),
			)
		}
	}

	if req.URL == "" {
		body, err := ioutil.ReadAll(env.Req.Body)
		if err != nil {
			return nil, qhttputil.NewError(http.StatusBadRequest, err.Error())
		}
		return s.Post(ctx, ProxyReq{
			Cmd:     req.Cmd,
			UID:     uint32(uid),
			ReqBody: body,
		}, env)
	}
	return s.Post(ctx, ProxyReq{
		Cmd: req.Cmd,
		UID: uint32(uid),
		URL: s.getDataURL(req.URL),
	}, env)
}

func (s service) getDataURL(_url string) string {
	return s.Host + "/data?url=" + url.QueryEscape(_url)
}
