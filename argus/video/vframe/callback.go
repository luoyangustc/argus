package vframe

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"strings"
	"sync"

	"github.com/qiniu/http/httputil.v1"
	"github.com/qiniu/rpc.v3"
	xlog "github.com/qiniu/xlog.v1"

	"qiniu.com/argus/atserving/model"
	URI "qiniu.com/argus/com/uri"
	"qiniu.com/argus/serving_eval"
	STS "qiniu.com/argus/sts/client"
)

type _ResponseCaller interface {
	SetHeader(error)
	Header() http.Header
	Call(context.Context, interface{}) error
}

var _ _ResponseCaller = &endResponse{}

type endResponse struct {
	header http.Header

	url   string
	id    string
	token string
	err   error

	*eval.AuditLog
}

func newEndResponse(url, id, token string, audit *eval.AuditLog) _ResponseCaller {
	return &endResponse{
		header:   make(http.Header),
		url:      url + "end/",
		id:       id,
		token:    token,
		AuditLog: audit,
	}
}

func (end *endResponse) SetHeader(err error) { end.err = err }
func (end *endResponse) Header() http.Header { return end.header }
func (end *endResponse) parseError(err error) (int, string) {
	if err == nil {
		return http.StatusOK, ""
	}
	switch err {
	default:
		return httputil.DetectError(err)
	}
}

func (end *endResponse) genMessage(resp interface{}) model.ResponseMessage {
	var (
		code, text = end.parseError(end.err)
		str, _     = json.Marshal(resp)
	)
	if end.AuditLog != nil {
		end.AuditLog.StatusCode = code
	}
	return model.ResponseMessage{
		ID:         end.id,
		StatusCode: code,
		StatusText: text,
		Header:     end.header,
		Response:   string(str),
	}
}

func (end *endResponse) Call(ctx context.Context, resp interface{}) error {
	var (
		xl  = xlog.FromContextSafe(ctx)
		msg = end.genMessage(resp)
		url = end.url + end.id + "/" + end.token
	)
	end.header[model.KEY_LOG] = xl.Xget()
	if end.AuditLog != nil {
		end.AuditLog.RespHeader = end.header
		end.AuditLog.RespBody = msg
	}
	err := rpc.DefaultClient.CallWithJson(ctx, nil, "POST", url, msg)
	xl.Infof("call: %s %#v %v", url, msg, err)
	return err
}

//----------------------------------------------------------------------------//

type _BeginClient interface {
	Post(context.Context) error
}

type beginClient struct {
	url   string
	id    string
	token string
}

func newBeginClient(url, id, token string) _BeginClient {
	return beginClient{
		url:   url + "begin/",
		id:    id,
		token: token,
	}
}

func (c beginClient) Post(ctx context.Context) error {
	var (
		xl  = xlog.FromContextSafe(ctx)
		url = c.url + c.id + "/" + c.token
	)

	err := rpc.DefaultClient.CallWithJson(ctx, nil, "POST", url, struct{}{})
	xl.Infof("call: %s %#v %v", url, err)
	return err
}

//----------------------------------------------------------------------------//

type _CutHandler interface {
	URI.Handler
	Del(context.Context, URI.Request) error
}

type cutHandler struct {
	URI.Handler
}

func newCutHandler() _CutHandler {
	return cutHandler{Handler: URI.WithFileHandler()}
}

func (h cutHandler) Del(ctx context.Context, req URI.Request) error {
	switch {
	case strings.HasPrefix(req.URI, "file://"):
		return os.Remove(strings.TrimPrefix(req.URI, "file://"))
	}
	return nil
}

type _CutClient interface {
	Post(context.Context, CutResponse) error
}

type cutClient struct {
	url   string
	id    string
	token string

	_CutHandler
	STS.Client
}

type CutClientConfig struct {
	// TODO
}

func newCutClient(
	url, id, token string,
	cutHandler _CutHandler,
	stsClient STS.Client,
	conf CutClientConfig,
) _CutClient {
	return cutClient{
		url:         url + "cuts/",
		id:          id,
		token:       token,
		_CutHandler: cutHandler,
		Client:      stsClient,
	}
}

func (c cutClient) Post(ctx context.Context, resp CutResponse) error {
	var (
		xl  = xlog.FromContextSafe(ctx)
		url = c.url + c.id + "/" + c.token
	)

	{
		var (
			wg   = new(sync.WaitGroup)
			done = func(err error) { wg.Done() }
			err  error
		)

		resp.Result.Cut.URI, err =
			func(ctx context.Context, uri string) (string, error) {
				_resp, err := c._CutHandler.Get(ctx, URI.Request{URI: uri})
				if err != nil {
					return "", err
				}
				_uri, _ := c.Client.NewURL(ctx, &_resp.Size)
				wg.Add(1)
				go func(ctx context.Context, rc io.ReadCloser) {
					defer func() {
						rc.Close()
						_ = c._CutHandler.Del(ctx, URI.Request{URI: uri})
					}()
					_ = c.Client.SyncPost(ctx, _uri, _resp.Size, rc, done)
				}(
					xlog.NewContext(ctx, xlog.FromContextSafe(ctx).Spawn()),
					_resp.Body,
				)
				return _uri, nil
			}(ctx, resp.Result.Cut.URI)
		if err != nil {
			return err
		}
		wg.Wait()
	}

	bs, _ := json.Marshal(resp)
	err := rpc.DefaultClient.CallWithJson(ctx, nil, "POST", url,
		model.ResponseMessage{Response: string(bs)},
	)
	xl.Infof("call: %s %#v %v", url, resp, err)
	return err
}
