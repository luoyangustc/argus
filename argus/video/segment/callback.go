package segment

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

type beginClientConfig struct {
	// TODO
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

type _ClipHandler interface {
	URI.Handler
	Del(context.Context, URI.Request) error
}

type clipHandler struct {
	URI.Handler
}

func newClipHandler() _ClipHandler {
	return clipHandler{Handler: URI.WithFileHandler()}
}

func (h clipHandler) Del(ctx context.Context, req URI.Request) error {
	switch {
	case strings.HasPrefix(req.URI, "file:///"):
		return os.Remove(strings.TrimPrefix(req.URI, "file://"))
	}
	return nil
}

type _ClipClient interface {
	Post(context.Context, ClipResponse) error
}

type clipClient struct {
	url   string
	id    string
	token string

	_ClipHandler
	STS.Client
}

type ClipClientConfig struct {
	// TODO
}

func newClipClient(
	url, id, token string,
	clipHandler _ClipHandler,
	stsClient STS.Client,
	conf ClipClientConfig,
) _ClipClient {
	return clipClient{
		url:          url + "clips/",
		id:           id,
		token:        token,
		_ClipHandler: clipHandler,
		Client:       stsClient,
	}
}

func (c clipClient) Post(ctx context.Context, resp ClipResponse) error {
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

		resp.Result.Clip.URI, err =
			func(ctx context.Context, uri string) (string, error) {
				_resp, err := c._ClipHandler.Get(ctx, URI.Request{URI: uri})
				if err != nil {
					return "", err
				}
				_uri, _ := c.Client.NewURL(ctx, &_resp.Size)
				wg.Add(1)
				go func(ctx context.Context, rc io.ReadCloser) {
					defer func() {
						rc.Close()
						c._ClipHandler.Del(ctx, URI.Request{URI: uri})
					}()
					_ = c.Client.SyncPost(ctx, _uri, _resp.Size, rc, done)
				}(
					xlog.NewContext(ctx, xlog.FromContextSafe(ctx).Spawn()),
					_resp.Body,
				)
				return _uri, nil
			}(ctx, resp.Result.Clip.URI)
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
