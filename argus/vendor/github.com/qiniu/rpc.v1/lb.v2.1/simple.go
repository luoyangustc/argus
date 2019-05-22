package lb

import (
	"io/ioutil"
	"net/http"
	"net/url"
	"time"

	cc "github.com/qiniu/io"
	"github.com/qiniu/rpc.v1"
	"github.com/qiniu/xlog.v1"
)

type simple struct {
	*Config
	client *rpc.Client
	sel    *selector
}

var DefaultTransport = NewTransport(nil)

func newSimple(cfg *Config, tr http.RoundTripper) *simple {
	if cfg.FailRetryIntervalS == 0 {
		cfg.FailRetryIntervalS = DefaultFailRetryInterval
	}
	if cfg.ShouldRetry == nil {
		cfg.ShouldRetry = ShouldRetry
	}
	if cfg.MaxFails == 0 {
		cfg.MaxFails = 1
	}
	if cfg.MaxFailsPeriodS == 0 {
		cfg.MaxFailsPeriodS = 1
	}
	if tr == nil {
		tr = DefaultTransport
	}

	client := &http.Client{
		Transport: tr,
		Timeout:   time.Duration(cfg.ClientTimeoutMS) * time.Millisecond,
	}
	return &simple{
		Config: cfg,
		client: &rpc.Client{client},
		sel:    newSelector(cfg.Hosts, cfg.TryTimes, cfg.FailRetryIntervalS, cfg.DnsResolve, cfg.DnsCacheTimeS, cfg.LookupHost, cfg.MaxFails, cfg.MaxFailsPeriodS),
	}
}

func (p *simple) doWithHostRet(req *Request) (rhost string, resp *http.Response, code int, err error) {

	ctx := req.Context()
	xl := xlog.FromContextSafe(ctx)

	reqURI := req.URL.RequestURI()
	httpreq := req.Request

	xl.Debug("simple.DoWithHostRet: start", reqURI)
	defer func() { xl.Debug("simple.DoWithHostRet: done", err) }()

	h, sel := p.sel.Get(xl)
	if h == nil {
		err = ErrServiceNotAvailable
		xl.Error("simple.DoWithHostRet: get host", err)
		return
	}
	rhost = h.raw

	tryTimes := p.sel.GetTryTimes()
	for i := uint32(0); i < tryTimes; i++ {

		httpreq.URL, err = url.Parse(rhost + reqURI)
		if err != nil {
			return
		}
		if req.Host == "" {
			if h.host != "" && p.DnsResolve && !p.LookupHostNotHoldHost {
				httpreq.Host = h.host
			} else {
				httpreq.Host = httpreq.URL.Host

				// rollback to raw host, such as c.host = "www.google.com"
				if httpreq.Host == "" {
					httpreq.Host = rhost
				}
			}
		}
		if req.Body != nil {
			httpreq.Body = ioutil.NopCloser(&cc.Reader{req.Body, 0})
		} else {
			httpreq.Body = nil
		}
		xl.Debug("simple.DoWithHostRet: with host", rhost+reqURI, httpreq.Host)
		resp, err = p.client.Do(xl, &httpreq)
		code = 0
		if resp != nil {
			code = resp.StatusCode
		}

		select {
		case <-ctx.Done():
			xl.Info("request canceled", err)
			return
		default:
		}

		if p.ShouldRetry(code, err) {
			xl.Warn("simple.DoWithHostRet: retry host, times: ", i, "code: ", code, "err: ", err, "host:", httpreq.URL.String())

			h.SetFail(xl)
			h = sel.Get(xl)
			if h == nil {
				xl.Error("simple.DoWithHostRet: get retry host", ErrServiceNotAvailable)
			} else {
				rhost = h.raw
			}

			// 这时候不会再去重试，不能关闭 resp.Body
			if h == nil || i == tryTimes-1 {
				xl.Debug("simple.DoWithHostRet: no more try", h, i)
				return
			}
			if resp != nil {
				discardAndClose(resp.Body)
			}
			continue
		}

		resp.Body = newBodyReader(xl, p.SpeedLimit, resp.Body, h)
		return
	}
	return
}
