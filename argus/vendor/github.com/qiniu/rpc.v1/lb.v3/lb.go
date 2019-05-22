// 重试逻辑:
// 发起请求，在配置的时间内未返回，发起重试请求。
// 重试请求不会打断之前已经发出的请求，最终使用最先返回的请求。
package lb

import (
	"io"
	"io/ioutil"
	"net/http"
	"strconv"
	"sync/atomic"
	"time"

	"code.google.com/p/go.net/context"

	"github.com/qiniu/http/httputil.v1"
	qio "github.com/qiniu/io"
	"github.com/qiniu/rpc.v1"
	"github.com/qiniu/xlog.v1"
)

var ErrCanceled = httputil.NewError(499, "canceled")

// --------------------------------------------------------------------
// func ShouldRetry
// default: retry network error and 5xx code

func ShouldRetry(code int, err error) bool {

	if err != nil {
		return true
	}
	return code/100 == 5
}

// ----------------------------------------------------------------------------

type Config struct {
	Http           *http.Client
	Hosts          []string
	HostRetrys     int
	RetryTimeoutMs int
	ShouldRetry    func(int, error) bool
}

type Client struct {
	client       rpc.Client
	hosts        []string
	hostTrys     int
	hostIndex    uint32
	retryTimeout time.Duration
	shouldRetry  func(int, error) bool
}

func New(cfg *Config) *Client {

	client := rpc.DefaultClient
	if cfg.Http != nil {
		client = rpc.Client{Client: cfg.Http}
	}
	hostTrys := cfg.HostRetrys + 1
	if n := len(cfg.Hosts); hostTrys > n {
		hostTrys = n
	}
	retryTimeout := time.Duration(cfg.RetryTimeoutMs) * time.Millisecond
	if retryTimeout == 0 {
		retryTimeout = 2 * time.Second
	}
	shouldRetry := cfg.ShouldRetry
	if shouldRetry == nil {
		shouldRetry = ShouldRetry
	}
	return &Client{
		client:       client,
		hosts:        cfg.Hosts,
		hostTrys:     hostTrys,
		retryTimeout: retryTimeout,
		shouldRetry:  shouldRetry,
	}
}

// -----------------------------------------------------------------------------

func discardAndClose(rc io.ReadCloser) {

	io.CopyN(ioutil.Discard, rc, 1024)
	rc.Close()
}

type requestRet struct {
	Logger *xlog.Logger
	Req    *http.Request
	Resp   *http.Response
	Err    error
}

func (c *Client) transport() http.RoundTripper {

	if c.client.Transport != nil {
		return c.client.Transport
	}
	return http.DefaultTransport
}

func (c *Client) Do(ctx context.Context, method, path, bodyType string, body io.ReaderAt, bodyLength int) (resp *http.Response, err error) {

	xl := xlog.FromContextSafe(ctx)

	var timeouts int
	var retryHosts []string
	retC := make(chan *requestRet, c.hostTrys)
	sends := make([]*http.Request, 0, c.hostTrys)
	recvs := make([]*http.Request, 0, c.hostTrys)

	index := int(atomic.AddUint32(&c.hostIndex, 1) % uint32(len(c.hosts)))
	host := c.hosts[index]
	client := c.client
forLoop:
	for {
		var r io.Reader
		if body != nil {
			r = &qio.Reader{ReaderAt: body}
		}
		req, err1 := http.NewRequest(method, host+path, r)
		if err1 != nil {
			// unexpected failed
			err = err1
			break
		}
		if bodyType != "" {
			req.Header.Set("Content-Type", bodyType)
		}
		req.ContentLength = int64(bodyLength)
		go func(xl *xlog.Logger, req *http.Request) {
			resp, err := client.Do(xl, req)
			retC <- &requestRet{xl, req, resp, err}
		}(xl.Spawn(), req)
		sends = append(sends, req)

	waitSelect:
		select {
		case <-ctx.Done():
			err = ctx.Err()
			if err == context.Canceled {
				err = ErrCanceled
			}
			xl.Xlog("lb.cancel")
			// interrupt in advance
			break forLoop
		case <-time.After(c.retryTimeout):
			timeouts++
			if len(sends) == c.hostTrys {
				xl.Warn("client.Do: timer expired, wait...")
				goto waitSelect
			}
			xl.Warn("Client.Do: timer expired, retry...")
		case ret := <-retC:
			resp, err = ret.Resp, ret.Err
			xl.Xput(ret.Logger.Xget())
			recvs = append(recvs, ret.Req)
			if len(recvs) == c.hostTrys {
				// exhaust all retrys
				break forLoop
			}
			var code int
			if err == nil {
				code = resp.StatusCode
			}
			if !c.shouldRetry(code, err) {
				// got an expected response
				break forLoop
			}
			if err == nil {
				discardAndClose(resp.Body)
			}
			xl.Warnf("Client.Do: call %v %v, code %v err %v", ret.Req.Host, path, code, err)
			if len(sends) == c.hostTrys {
				goto waitSelect
			}
		}
		if retryHosts == nil {
			retryHosts = copyExcept(c.hosts, index)
		}
		retryHosts, host = randomShrink(retryHosts)
	}
	if timeouts > 0 {
		xl.Xlog("lb.timeouts:" + strconv.Itoa(timeouts))
	}

	if len(recvs) < len(sends) {
		go func() {
			type canceler interface {
				CancelRequest(*http.Request)
			}
			tr, ok := c.transport().(canceler)
			if ok {
				for _, req := range sends {
					if pos := indexRequest(recvs, req); pos == -1 {
						tr.CancelRequest(req)
					}
				}
			} else {
				xl.Warnf("Client.Do: type %T doesn't support CancelRequest", c.transport())
			}
			waits := len(sends) - len(recvs)
			for i := 0; i < waits; i++ {
				ret := <-retC
				var code int
				if ret.Err == nil {
					code = ret.Resp.StatusCode
					discardAndClose(ret.Resp.Body)
				}
				xl.Warnf("Client.Do: wait call %v %v, code %v, err %v", ret.Req.Host, path, code, ret.Err)
			}
		}()
	}
	return
}

// -----------------------------------------------------------------------------

func (c *Client) CallWith(
	ctx context.Context, ret interface{}, path string, bodyType string, body io.ReaderAt, bodyLength int) (err error) {

	resp, err := c.PostWith(ctx, path, bodyType, body, bodyLength)
	if err != nil {
		return err
	}
	l := xlog.FromContextSafe(ctx)
	return rpc.CallRet(l, ret, resp)
}

func (c *Client) PostWith(
	ctx context.Context, path, bodyType string, body io.ReaderAt, bodyLength int) (resp *http.Response, err error) {

	return c.Do(ctx, "POST", path, bodyType, body, bodyLength)
}
