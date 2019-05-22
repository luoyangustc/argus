// +build go1.7

package rpc

import (
	"context"
	"io"
	"net/http"
	"sync"

	"github.com/qiniu/xlog.v1"
)

// 上游取消请求，下游要及时结束
func (r *Client) doCtx(l Logger, req *http.Request) (resp *http.Response, err error) {

	xl := xlog.NewWith(l)
	upCtx := xl.Context()
	// 确保如果上游已经关闭了，那么请求不会打到客户端。see: https://jira.qiniu.io/browse/KODO-4243
	select {
	case <-upCtx.Done():
		xl.Info("wrap err with code 499", upCtx.Err())
		return nil, NewHttpError(499, upCtx.Err())
	default:
	}
	// 将req而不是xl的context传下去，因为req中可能含有ClientTrace。比如:包authrpc中用到了
	ctx, cancel := context.WithCancel(req.Context())
	req = req.WithContext(ctx)

	done := make(chan struct{})
	go func() {
		select {
		case <-upCtx.Done():
			cancel()
		case <-done:
		}
	}()
	resp, err = r.Client.Do(req)

	var once sync.Once
	stop := func() { once.Do(func() { close(done) }) }
	wrapErr := func(err error) error {
		select {
		case <-upCtx.Done():
			xl.Info("wrap err with code 499", err)
			return NewHttpError(499, err)
		default:
			return err
		}
	}
	if err == nil {
		// If the returned error is nil, the Response will contain a non-nil Body
		resp.Body = &cancelBody{
			rc:      resp.Body,
			wrapErr: wrapErr,
			stop:    stop,
		}
	} else {
		// On error, any Response can be ignored
		// go routine can be closed already, so close instead of send
		stop()
		err = wrapErr(err)
	}
	return
}

// copy from lb.v2.1/ctx.go
type cancelBody struct {
	stop    func()
	wrapErr func(err error) error
	rc      io.ReadCloser
}

func (b *cancelBody) Read(p []byte) (n int, err error) {
	n, err = b.rc.Read(p)
	// never wrap EOF error
	if err == nil || err == io.EOF {
		return n, err
	}
	b.stop()
	// if context cancel is caused by upCtx, wrap 499 code
	err = b.wrapErr(err)
	return n, err
}

func (b *cancelBody) Close() error {
	err := b.rc.Close()
	b.stop()
	return err
}
