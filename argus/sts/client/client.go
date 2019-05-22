package client

import (
	"context"
	"io"
	"math"
	"net/http"
	"net/url"
	"strconv"
	"strings"

	"github.com/qiniu/http/httputil.v1"
	"github.com/qiniu/rpc.v3"
	"github.com/qiniu/xlog.v1"

	HTTP "qiniu.com/argus/com/http"
)

type getRequest struct {
	uri    string
	length *int64

	onlyProxy bool
	beginOff  *int64
	endOff    *int64
}

type GetOption func(*getRequest)

func WithOnlyProxy() GetOption { return func(req *getRequest) { req.onlyProxy = true } }
func WithRange(begin, end int64) GetOption {
	return func(req *getRequest) {
		req.beginOff = &begin
		if end != math.MaxInt64 {
			req.endOff = &end
		} else {
			req.endOff = nil
		}
	}
}

type URIOptions byte

const (
	_OPTION_NONE URIOptions = 0x00
	OPTION_SYNC  URIOptions = 0x01
	OPTION_PROXY URIOptions = 0x02
)

// Client ...
type Client interface {
	NewURL(context.Context, *int64) (string, error)
	GetURL(context.Context, string, *int64, *URIOptions) (string, error)

	DoFetch(context.Context, string, *int64, bool) (string, int64, error)
	Post(context.Context, string, int64, io.Reader) error
	SyncPost(context.Context, string, int64, io.Reader, func(error)) error
	Get(context.Context, string, *int64, ...GetOption) (io.ReadCloser, int64, http.Header, error)
}

const (
	SCHEME_STS  = "sts"
	SCHEME_HTTP = "http"
)

var _ Client = client{}

type client struct {
	rpc.Client
	host   string
	newKey func() string
}

// NewClient ...
func NewClient(host string, newKey func() string, c *rpc.Client) Client {
	if c == nil {
		c = &rpc.DefaultClient
	}
	return client{
		host:   host,
		newKey: newKey,
		Client: *c,
	}
}

func (c client) NewURL(ctx context.Context, length *int64) (string, error) {
	return "sts://" + c.host + "/v1/file/" + c.newKey(), nil
}

func (c client) GetURL(
	ctx context.Context, uri string, length *int64, options *URIOptions,
) (string, error) {
	if strings.HasPrefix(uri, "sts://") {
		return uri, nil
	}
	var r string
	if options != nil && *options&OPTION_PROXY == OPTION_PROXY {
		r = "sts://" + c.host + "/v1/proxy?uri=" + url.QueryEscape(uri)
	} else {
		r = "sts://" + c.host + "/v1/fetch?uri=" + url.QueryEscape(uri)
		if options != nil && *options&OPTION_SYNC == OPTION_SYNC {
			r += "&sync=true"
		}
	}
	if length != nil {
		r += "&length=" + strconv.FormatInt(*length, 10)
	}
	return r, nil
}

func (c client) DoFetch(ctx context.Context, uri string, length *int64, sync bool) (string, int64, error) {
	var options = _OPTION_NONE | OPTION_SYNC
	uri, _ = c.GetURL(ctx, uri, length, &options)
	uri2 := strings.Replace(uri, SCHEME_STS, SCHEME_HTTP, 1)
	ret := struct {
		Length int64 `json:"length"`
	}{}
	if err := c.Call(ctx, &ret, "POST", uri2); err != nil {
		return uri, 0, err
	}
	if length == nil {
		return uri, ret.Length, nil
	}
	return uri, *length, nil
}

func (c client) Post(ctx context.Context, uri string, length int64, r io.Reader) error {
	uri = strings.Replace(uri, SCHEME_STS, SCHEME_HTTP, 1)
	uri = uri + "?length=" + strconv.FormatInt(length, 10)
	if err := c.CallWith64(
		ctx, nil,
		"POST", uri, "application/octet-stream",
		r, length,
	); err != nil {
		return nil
	}
	return nil
}

func (c client) SyncPost(
	ctx context.Context,
	uri string,
	length int64,
	r io.Reader,
	syncDone func(error),
) error {

	xl := xlog.FromContextSafe(ctx)

	uri, host, err := openURI(ctx, uri)
	if err != nil {
		syncDone(err)
		xl.Warnf("open uri failed. %s %v", uri, err)
		return err
	}
	uri = uri + "?length=" + strconv.FormatInt(length, 10)
	var id = struct {
		ID string `json:"id"`
	}{}
	if err := c.CallWithForm(
		ctx, &id, "POST", uri,
		map[string][]string{"length": []string{strconv.FormatInt(length, 10)}},
	); err != nil {
		syncDone(err)
		xl.Warnf("open sts failed. %s %v", uri, err)
		return err
	}

	syncDone(nil)

	uri = "http://" + host + "/v1/write/" + id.ID
	if err := c.CallWith64(
		ctx, nil,
		"POST", uri, "application/octet-stream",
		r, length,
	); err != nil {
		xl.Warnf("write sts faield. %s %v", uri, err)
		return err
	}
	xl.Infof("write sts done. %s", uri)
	return nil
}

func (c client) Get(ctx context.Context, uri string, length *int64, options ...GetOption,
) (io.ReadCloser, int64, http.Header, error) {
	req := getRequest{uri: uri, length: length}
	for _, opt := range options {
		opt(&req)
	}
	if req.onlyProxy {
		var options = _OPTION_NONE | OPTION_SYNC
		req.uri, _ = c.GetURL(ctx, req.uri, req.length, &options)
	} else {
		req.uri, _ = c.GetURL(ctx, req.uri, req.length, nil)
	}
	req.uri = strings.Replace(req.uri, SCHEME_STS, SCHEME_HTTP, 1)
	req0, err := http.NewRequest("GET", req.uri, nil)
	if err != nil {
		return nil, 0, nil, err
	}
	if req.beginOff != nil {
		HTTP.FormatRangeRequest(req0.Header, req.beginOff, req.endOff)
	}
	resp, err := c.Do(ctx, req0)
	if err != nil {
		return nil, 0, nil, err
	}
	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		err = rpc.ResponseError(resp)
		return nil, 0, nil, httputil.NewError(resp.StatusCode, err.Error())
	}
	return resp.Body, resp.ContentLength, resp.Header, nil
}

//----------------------------------------------------------------------------//

func openURI(ctx context.Context, uri string) (string, string, error) {
	_url, err := url.Parse(uri)
	if err != nil {
		return "", "", err
	}
	_url.Path = strings.Replace(_url.Path, "/v1/file", "/v1/open", 1)
	ret := _url.String()
	return strings.Replace(ret, SCHEME_STS, SCHEME_HTTP, 1), _url.Host, nil
}
