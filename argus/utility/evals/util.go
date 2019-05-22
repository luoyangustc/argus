package evals

import (
	"context"
	"io/ioutil"
	"net/http"
	"time"

	"github.com/qiniu/rpc.v3"
	xlog "github.com/qiniu/xlog.v1"

	ahttp "qiniu.com/argus/argus/com/http"
)

type SimpleReq struct {
	Data struct {
		URI string `json:"uri"`
	} `json:"data"`
}

////////////////////////////////////////////////////////////////////////////////

func callRetry(ctx context.Context, f func(context.Context) error) error {
	return ahttp.CallWithRetry(ctx, []int{530}, []func(context.Context) error{f, f})
}

//----------------------------------------------------------------------------//

type SimpleEval interface {
	Eval(ctx context.Context, uid, utype uint32, req, ret interface{}) error
}

func NewSimpleEval(host, path string, timeout time.Duration) SimpleEval {
	return _Simple{host: host, path: path, timeout: timeout}
}

type _Simple struct {
	host    string
	path    string
	timeout time.Duration
}

func (s _Simple) Eval(
	ctx context.Context, uid, utype uint32,
	req, ret interface{},
) error {
	var (
		client = ahttp.NewQiniuStubRPCClient(uid, utype, s.timeout)
	)
	return callRetry(ctx,
		func(ctx context.Context) error {
			return client.CallWithJson(ctx, ret, "POST", s.host+s.path, req)
		})
}

type SimpleBinEval interface {
	Eval(ctx context.Context, uid, utype uint32, req interface{}) ([]byte, error)
}

func NewSimpleBinEval(host, path string, timeout time.Duration) SimpleBinEval {
	return _SimpleBin{host: host, path: path, timeout: timeout}
}

type _SimpleBin struct {
	host    string
	path    string
	timeout time.Duration
}

func (s _SimpleBin) Eval(
	ctx context.Context, uid, utype uint32, req interface{},
) (bs []byte, err error) {

	var (
		xl     = xlog.FromContextSafe(ctx)
		client = ahttp.NewQiniuStubRPCClient(uid, utype, s.timeout)
	)
	var resp *http.Response
	err = callRetry(ctx,
		func(ctx context.Context) error {
			var err1 error

			resp, err1 = client.DoRequestWithJson(ctx, "POST", s.host+s.path, req)
			return err1
		})
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode/100 != 2 || resp.ContentLength == 0 {
		xl.Errorf(
			"call "+s.host+s.path+" error:%v,status code:%v,content length:%v,req:%v",
			err, resp.StatusCode, resp.ContentLength, req,
		)
		return nil, rpc.ResponseError(resp)
	}
	bs, err = ioutil.ReadAll(resp.Body)
	if err != nil {
		xl.Errorf("call "+s.host+s.path+",read resp body error:%v", err)
	}
	return bs, err
}
