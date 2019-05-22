package lb

import (
	"errors"
	"net"
	"net/http"
	"net/url"
	"time"

	"sync/atomic"

	"github.com/qiniu/http/httputil.v1"
	"github.com/qiniu/log.v1"
	"github.com/qiniu/rpc.v1"
	"github.com/qiniu/xlog.v1"
)

// --------------------------------------------------------------------

var ShouldReproxy = func(code int, err error) bool {
	if code == 503 {
		return true // nginx 在超过连接数限制时返回 503
	}
	if err == nil {
		return false // 成功
	}
	if _, ok := err.(rpc.RespError); ok {
		return false // 不需要重试的返回码
	}
	return true
}

// --------------------------------------------------------------------

type TransportConfig struct {
	DialTimeoutMS       int  `json:"dial_timeout_ms"` // 默认 2s
	RespTimeoutMS       int  `json:"resp_timeout_ms"`
	MaxIdleConnsPerHost int  `json:"max_idle_conns_per_host"`
	DisableCompression  bool `json:"disable_compression"`

	// 代理的配置，地址为空时忽略
	Proxys             []string `json:"proxys"`
	TryTimes           uint32   `json:"try_times"`
	FailRetryIntervalS int64    `json:"fail_retry_interval_s"`
	DnsResolve         bool     `json:"dns_resolve"`
	DnsCacheTimeS      int64    `json:"dns_cache_time_s"`
	MaxFails           int      `json:"max_fails"`          //默认 1
	MaxFailsPeriodS    int64    `json:"max_fails_period_s"` //默认为1s, 即对于一个host，在MaxFailsPeriodS时间内，失败的数量大于等于MaxFails, 认为线路断开

	ShouldReproxy func(code int, err error) bool      `json:"-"`
	LookupHost    func(host string) ([]string, error) `json:"-"`

	Dial func(network, addr string) (net.Conn, error) `json:"-"`
}

type Transport struct {
	*TransportConfig
	sel    *selector
	tr     *http.Transport
	closed int32
}

func NewTransport(cfg *TransportConfig) http.RoundTripper {
	if cfg == nil {
		cfg = &TransportConfig{}
	}

	if cfg.DialTimeoutMS == 0 {
		cfg.DialTimeoutMS = DefaultDialTimeoutMS
	}
	if cfg.FailRetryIntervalS == 0 {
		cfg.FailRetryIntervalS = DefaultFailRetryInterval
	}
	if cfg.ShouldReproxy == nil {
		cfg.ShouldReproxy = ShouldReproxy
	}
	if cfg.MaxFails == 0 {
		cfg.MaxFails = 1
	}
	if cfg.MaxFailsPeriodS == 0 {
		cfg.MaxFailsPeriodS = 1
	}
	tr := &http.Transport{
		Proxy: http.ProxyFromEnvironment,
		ResponseHeaderTimeout: time.Duration(cfg.RespTimeoutMS) * time.Millisecond,
		TLSHandshakeTimeout:   10 * time.Second,
		MaxIdleConnsPerHost:   cfg.MaxIdleConnsPerHost,
		DisableCompression:    cfg.DisableCompression,
		Dial:                  cfg.Dial,
	}
	if tr.Dial == nil {
		tr.Dial = (&net.Dialer{
			Timeout:   time.Duration(cfg.DialTimeoutMS) * time.Millisecond,
			KeepAlive: 30 * time.Second,
		}).Dial
	}
	var sel *selector
	if len(cfg.Proxys) > 0 {
		sel = newSelector(cfg.Proxys, cfg.TryTimes, cfg.FailRetryIntervalS, cfg.DnsResolve, cfg.DnsCacheTimeS, cfg.LookupHost, cfg.MaxFails, cfg.MaxFailsPeriodS)
		tr.Proxy = func(req *http.Request) (u *url.URL, err error) {
			h, ok := sel.GetReqHost(req)
			if !ok {
				err = errors.New("can not find proxy host for invalid request")
			} else {
				u = h.URL
			}
			return
		}
	}

	if sel != nil {
		log.Debug("NewTransport with proxys", cfg.Proxys)
	}

	return &Transport{TransportConfig: cfg, sel: sel, tr: tr}
}

// 代理错误，客户端不需要重试目标服务端
type ProxyError struct{ error }

func (t *Transport) RoundTrip(req *http.Request) (resp *http.Response, err error) {
	xl := xlog.NewWithReq(req)

	xl.Debug("RoundTrip: start", req.URL.RequestURI())
	defer func() { xl.Debug("RoundTrip: done", err) }()

	// 没有配置代理，直接请求返回
	if t.sel == nil {
		resp, err = t.tr.RoundTrip(req)
		return
	}

	proxy, sel := t.sel.Get(xl)
	if proxy == nil {
		err = &ProxyError{ErrServiceNotAvailable}
		xl.Error("get proxy failed:", err)
		return
	}

	defer t.sel.DelReqHost(req) // 释放该请求

	tryTimes := t.sel.GetTryTimes()
	for i := uint32(0); i < tryTimes; i++ {
		xl.Debug("RoundTrip: with proxy", proxy.raw)
		t.sel.SetReqHost(req, proxy) // 请求之前设置 proxy host 给回调函数 Proxy 使用

		resp, err = t.tr.RoundTrip(req)
		code := 0
		if resp != nil {
			code = resp.StatusCode
		}
		if err != nil {
			err = &ProxyError{err}
		}

		if t.ShouldReproxy(code, err) {
			xl.Warn("retry proxy, times: ", i, "code: ", code, "err: ", err)

			proxy.SetFail(xl)
			proxy = sel.Get(xl)
			if proxy == nil {
				xl.Error("get retry proxy failed:", ErrServiceNotAvailable)
			}

			// 这时候不会再去重试，不能关闭 resp.Body
			if proxy == nil || i == tryTimes-1 {
				xl.Debug("no more try", proxy, i)
				return
			}
			if resp != nil {
				discardAndClose(resp.Body)
			}
			continue
		}
		return
	}

	if atomic.LoadInt32(&t.closed) != 0 {
		t.tr.CloseIdleConnections()
	}
	return
}

func (t *Transport) Close() error {
	atomic.StoreInt32(&t.closed, 1)
	t.tr.CloseIdleConnections()
	return nil
}

func (t *Transport) CancelRequest(req *http.Request) {
	rc, ok := httputil.GetRequestCanceler(t.tr)
	if ok {
		rc.CancelRequest(req)
	} else {
		xl := xlog.NewWithReq(req)
		xl.Panic("can not get canceler, cancal request failed")
	}
}
