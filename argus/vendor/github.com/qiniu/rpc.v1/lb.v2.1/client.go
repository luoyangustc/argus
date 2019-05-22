// 在 lb.v2 的基础上进行了修改，可定制客户端和服务端的轮询策略，更好的支持正向代理
// TODO 对 body 传输失败进行重试
package lb

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/url"
	"strings"

	"code.google.com/p/go.net/context"
	cc "github.com/qiniu/io"
	"github.com/qiniu/log.v1"
	"github.com/qiniu/rpc.v1"
	"github.com/qiniu/xlog.v1"
)

const (
	DefaultFailRetryInterval = -1 // 默认关闭
	DefaultDialTimeoutMS     = 2000
	DefaultDnsCacheTimeS     = 15 * 60 // 如果开启dns查询, dns 缓存时间
)

// 不要使用 http error，否则会被判定为 rpc.RespError
var ErrServiceNotAvailable = errors.New("service not available")

// --------------------------------------------------------------------
var ShouldRetry = func(code int, err error) bool {
	// 使用代理时会返回 50x，需要重试目标服务，但是不更换代理。（假设代理总能连上一个目标服务）
	if code == 502 || code == 504 {
		return true // 服务出错
	}
	if err == nil {
		return false // 成功
	}
	if _, ok := err.(*ProxyError); ok {
		return false // 不需要重试代理错误
	}
	if _, ok := err.(rpc.RespError); ok {
		return false // 不需要重试的返回码
	}
	return true
}

var ShouldFailover = func(code int, err error) bool {
	return ShouldRetry(code, err) || ShouldReproxy(code, err)
}

// --------------------------------------------------------------------
// type Request
type Request struct {
	http.Request
	Body io.ReaderAt

	ctx context.Context // support go1.5+
}

func NewRequest(method, urlStr string, body io.ReaderAt) (*Request, error) {

	var r io.Reader
	if body != nil {
		r = &cc.Reader{body, 0}
	}
	httpreq, err := http.NewRequest(method, urlStr, r)
	if err != nil {
		return nil, err
	}
	req := &Request{*httpreq, body, context.Background()}
	return req, nil
}

func (r *Request) Context() context.Context {
	if r.ctx != nil {
		return r.ctx
	}
	return context.Background()
}

// --------------------------------------------------------------------

// client 和 failover 各自拥有独立的 Config 适配不同的环境。
// client 和 failover 内部重试独立，failover 使用场景是 client 重试全部失败了才会使用 failover，用于实现优先使用 client，failover 作为备用的逻辑。
type Config struct {
	Hosts              []string `json:"hosts"`                 // 目标服务地址列表，不能为空
	TryTimes           uint32   `json:"try_times"`             // 对于某个请求 client 或者 failover 各自最大尝试次数，TryTimes = 重试次数 + 1
	FailRetryIntervalS int64    `json:"fail_retry_interval_s"` // 默认为 -1 关闭。当 > 0 时，表示将某个失败的 host 屏蔽的时间，屏蔽期间该 host 不会被使用，超过该时间后该 host 恢复正常。建议只在配置有 failover 时启用，达到快速转换到备用线路的效果
	ClientTimeoutMS    int      `json:"client_timeout_ms"`     // http 客户端超时时间，见 http.Client
	DnsResolve         bool     `json:"dns_resolve"`           // 开启 DNS 解析实现 DNS 负载均衡
	DnsCacheTimeS      int64    `json:"dns_cache_time_s"`      // DNS 缓存时间
	MaxFails           int      `json:"max_fails"`             // 默认 1，只在 FailRetryIntervalS > 0 时有效。类似 nginx 的 max_fails，表示某一个 host 失败几次之后被加入到屏蔽列表
	MaxFailsPeriodS    int64    `json:"max_fails_period_s"`    // 默认为1s, 即对于一个host，在 MaxFailsPeriodS 时间内，失败的数量大于等于 MaxFails, 认为线路断开。

	SpeedLimit // 低速熔断配置，默认关闭，只在 FailRetryIntervalS > 0 时有效

	ShouldRetry func(code int, err error) bool                `json:"-"`
	LookupHost  func(host string) (addrs []string, err error) `json:"-"` // If nil, net.LookupHost is used. 需要DnsResolve为true才能生效

	// 在使用LookupHost的时候, 有一个正常的需求是保留原来的host
	// 例如: http//a.com/xxxx --> http//1.1.1.1/xxx -H 'host:a.com'
	// 如果这个请求使用lb.v2.1.(*Transport)的代理, 请求变成
	// http//1.1.1.1/xxx -H 'host:a.com' -x 'b.com'
	// 但是,这个请求不符合http协议,并且在go语言中会把请求变成下面的格式发出去:
	// http//a.com/xxx -H 'host:a.com' -x 'b.com', LookupHost 就失去意义了。
	// 某些场合又是必须使用代理, 比如跨机房的公网重试。
	// 因此, LookupHost是否保留host的开关似乎是必须的。

	// 如果有下个版本的lb.v2.2的话, 希望可以改进LookupHost 变成
	// func(host string) (host string, addrs []string, err error)
	LookupHostNotHoldHost bool `json:"lookup_host_not_hold_host"`
}

type Client struct {
	client         *simple
	failover       *simple
	shouldFailover func(int, error) bool
}

func New(cfg *Config, tr http.RoundTripper) *Client {
	return NewWithFailover(cfg, nil, tr, nil, nil)
}

func NewWithFailover(client, failover *Config, clientTr, failoverTr http.RoundTripper, shouldFailover func(int, error) bool) *Client {
	if len(client.Hosts) == 0 {
		log.Panic("client hosts should not be empty")
	}

	p := &Client{client: newSimple(client, clientTr)}

	if failover != nil && len(failover.Hosts) > 0 {
		p.failover = newSimple(failover, failoverTr)

		if shouldFailover == nil {
			shouldFailover = ShouldFailover
		}
		p.shouldFailover = shouldFailover
	}
	return p
}

func (r Client) Get(l rpc.Logger, url string) (resp *http.Response, err error) {

	req, err := NewRequest("GET", url, nil)
	if err != nil {
		return
	}
	return r.Do(l, req)
}

func (r Client) GetCall(l rpc.Logger, ret interface{}, url1 string) (err error) {

	resp, err := r.Get(l, url1)
	if err != nil {
		return err
	}
	return rpc.CallRet(l, ret, resp)
}

func (r Client) DeleteCall(l rpc.Logger, ret interface{}, url1 string) (err error) {

	req, err := NewRequest("DELETE", url1, nil)
	if err != nil {
		return
	}
	resp, err := r.Do(l, req)
	if err != nil {
		return err
	}
	return rpc.CallRet(l, ret, resp)

}

func (p *Client) Do(
	l rpc.Logger, req *Request) (resp *http.Response, err error) {
	_, resp, err = p.DoWithHostRet(l, req)
	return
}

func (p *Client) CallWith64(
	l rpc.Logger, ret interface{}, path string, bodyType string, body io.ReaderAt, bodyLength int64) (err error) {

	resp, err := p.PostWith64(l, path, bodyType, body, bodyLength)
	if err != nil {
		return
	}
	return rpc.CallRet(l, ret, resp)
}

func (p *Client) CallWithForm(l rpc.Logger,
	ret interface{}, path string, params map[string][]string) (err error) {

	resp, err := p.PostWithForm(l, path, params)
	if err != nil {
		return
	}
	return rpc.CallRet(l, ret, resp)
}

func (p *Client) CallWithJson(l rpc.Logger,
	ret interface{}, path string, params interface{}) (err error) {

	resp, err := p.PostWithJson(l, path, params)
	if err != nil {
		return
	}
	return rpc.CallRet(l, ret, resp)
}

func (p *Client) CallWith(
	l rpc.Logger, ret interface{}, path string, bodyType string, body io.ReaderAt, bodyLength int) (err error) {

	resp, err := p.PostWith(l, path, bodyType, body, bodyLength)
	if err != nil {
		return err
	}
	return rpc.CallRet(l, ret, resp)
}

func (p *Client) Call(
	l rpc.Logger, ret interface{}, path string) (err error) {

	resp, err := p.PostEx(l, path)
	if err != nil {
		return err
	}
	return rpc.CallRet(l, ret, resp)
}

func (p *Client) PostEx(l rpc.Logger, path string) (resp *http.Response, err error) {
	req, err := NewRequest("POST", path, nil)
	if err != nil {
		return
	}
	return p.Do(l, req)
}

func (p *Client) PostWith64(
	l rpc.Logger, path, bodyType string, body io.ReaderAt, bodyLength int64) (resp *http.Response, err error) {
	req, err := NewRequest("POST", path, body)
	if err != nil {
		return
	}
	req.Header.Set("Content-Type", bodyType)
	req.ContentLength = bodyLength
	return p.Do(l, req)
}

func (p *Client) PostWith(
	l rpc.Logger, path, bodyType string, body io.ReaderAt, bodyLength int) (resp *http.Response, err error) {
	_, resp, err = p.PostWithHostRet(l, path, bodyType, body, bodyLength)
	return
}

func (p *Client) PostWithForm(
	l rpc.Logger, path string, params map[string][]string) (resp *http.Response, err error) {
	msg := url.Values(params).Encode()
	_, resp, err = p.PostWithHostRet(l, path, "application/x-www-form-urlencoded", strings.NewReader(msg), len(msg))
	return
}

func (p *Client) PostWithJson(
	l rpc.Logger, path string, params interface{}) (resp *http.Response, err error) {
	msg, err := json.Marshal(params)
	if err != nil {
		return
	}
	_, resp, err = p.PostWithHostRet(l, path, "application/json", bytes.NewReader(msg), len(msg))
	return
}

func (p *Client) PostWithHostRet(
	l rpc.Logger, path, bodyType string, body io.ReaderAt, bodyLength int) (host string, resp *http.Response, err error) {
	req, err := NewRequest("POST", path, body)
	if err != nil {
		return
	}
	req.Header.Set("Content-Type", bodyType)
	req.ContentLength = int64(bodyLength)
	return p.DoWithHostRet(l, req)
}

func (p *Client) DoCtxWithHostRet(
	req *Request) (host string, resp *http.Response, err error) {

	host, resp, code, err := p.client.doCtxWithHostRet(req)
	ctx := req.Context()
	xl := xlog.FromContextSafe(ctx)
	select {
	case <-ctx.Done():
		return
	default:
	}
	if p.failover == nil || !p.shouldFailover(code, err) {
		return
	}
	if resp != nil {
		discardAndClose(resp.Body)
	}
	xl.Warn("try failover client")
	host, resp, _, err = p.failover.doCtxWithHostRet(req)
	return
}
