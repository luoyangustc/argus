// +build go1.5

package lb

import (
	"io"
	"net/http"
	"sync"

	"github.com/qiniu/rpc.v1"
	"github.com/qiniu/xlog.v1"

	"code.google.com/p/go.net/context"
)

func (r *Request) WithContext(ctx context.Context) *Request {
	if ctx == nil {
		panic("nil context")
	}
	r2 := new(Request)
	*r2 = *r
	r2.ctx = ctx
	return r2
}

func (p *Client) CallWithCtx(
	ctx context.Context, ret interface{}, path string) (err error) {
	req, err := NewRequest("POST", path, nil)
	if err != nil {
		return
	}
	req = req.WithContext(ctx)
	resp, err := p.DoWithCtx(req)
	if err != nil {
		return err
	}
	return ctxCallRet(ctx, ret, resp)
}

func (p *Client) DoWithCtx(
	req *Request) (resp *http.Response, err error) {
	_, resp, err = p.DoCtxWithHostRet(req)
	return
}

func ctxCallRet(ctx context.Context, ret interface{}, resp *http.Response) (err error) {
	xl := xlog.FromContextSafe(ctx)
	return rpc.CallRet(xl, ret, resp)
}

func (p *Client) DoWithHostRet(
	l rpc.Logger, req *Request) (host string, resp *http.Response, err error) {

	xl := xlog.NewWith(l)
	req = req.WithContext(xlog.NewContext(req.Context(), xl))
	return p.DoCtxWithHostRet(req)
}

func (p *simple) doCtxWithHostRet(req *Request) (host string, resp *http.Response, code int, err error) {

	ctx := req.Context()
	xl := xlog.FromContextSafe(ctx)
	if xl != nil {
		req.Header.Set("X-Reqid", xl.ReqId())
	}
	stop := func() {}
	if initial := req.Cancel; initial != nil {
		var cancel context.CancelFunc
		ctx, cancel = context.WithCancel(req.Context())
		req = req.WithContext(ctx)

		done := make(chan struct{})
		var once sync.Once
		stop = func() { once.Do(func() { close(done) }) }

		go func() {
			select {
			case <-initial:
				cancel()
			case <-done:
				break
			}
		}()
	}
	req.Cancel = ctx.Done()
	host, resp, code, err = p.doWithHostRet(req)
	if resp != nil {
		resp.Body = &cancelBody{
			stop: stop,
			rc:   resp.Body,
		}
	}
	return
}

type cancelBody struct {
	stop func()
	rc   io.ReadCloser
}

func (b *cancelBody) Read(p []byte) (n int, err error) {
	n, err = b.rc.Read(p)
	if err == nil {
		return n, nil
	}
	b.stop()
	return n, err
}

func (b *cancelBody) Close() error {
	err := b.rc.Close()
	b.stop()
	return err
}
