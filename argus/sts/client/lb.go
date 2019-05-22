package client

import (
	"context"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"sync"

	"github.com/qiniu/http/httputil.v1"
	"github.com/qiniu/rpc.v3"
	"github.com/qiniu/xlog.v1"
	"qbox.us/dht"

	HTTP "qiniu.com/argus/com/http"
)

var _ Client = &lb{}

type lb struct {
	dht.Interface
	rpc.Client

	nodes  dht.NodeInfos
	newKey func() string

	*sync.RWMutex
}

// NewLB ...
func NewLB(nodes dht.NodeInfos, newKey func() string, c *rpc.Client) (
	Client, func(context.Context, dht.NodeInfos)) {

	if c == nil {
		c = &rpc.DefaultClient
	}
	_lb := &lb{
		Interface: dht.NewCarp(nodes),
		Client:    *c,
		nodes:     nodes,
		newKey:    newKey,
		RWMutex:   new(sync.RWMutex),
	}
	return _lb, _lb.update
}

func (lb *lb) update(ctx context.Context, nodes dht.NodeInfos) {
	lb.Lock()
	defer lb.Unlock()
	lb.Interface.Setup(nodes)
}

func (lb *lb) route(key []byte, ttl int) dht.RouterInfos {
	lb.RLock()
	defer lb.RUnlock()
	return lb.Route(key, ttl)
}

func (lb *lb) NewURL(ctx context.Context, length *int64) (string, error) {
	var (
		key   = lb.newKey()
		infos = lb.route([]byte(key), 1)
	)
	return "sts://" + infos[0].Host + "/v1/file/" + key, nil
}

func (lb *lb) GetURL(ctx context.Context, uri string, length *int64, options *URIOptions) (string, error) {
	if strings.HasPrefix(uri, "sts://") {
		return uri, nil
	}
	var (
		infos = lb.route([]byte(uri), 1)
		host  = infos[0].Host
	)
	var r string
	if options != nil && *options&OPTION_PROXY == OPTION_PROXY {
		r = "sts://" + host + "/v1/proxy?uri=" + url.QueryEscape(uri)
	} else {
		r = "sts://" + host + "/v1/fetch?uri=" + url.QueryEscape(uri)
		if options != nil && *options&OPTION_SYNC == OPTION_SYNC {
			r += "&sync=true"
		}
	}
	if length != nil {
		r += "&length=" + strconv.FormatInt(*length, 10)
	}
	return r, nil
}

func (lb *lb) DoFetch(ctx context.Context, uri string, length *int64, sync bool) (string, int64, error) {
	var options = _OPTION_NONE | OPTION_SYNC
	uri, _ = lb.GetURL(ctx, uri, length, &options)
	uri2 := strings.Replace(uri, SCHEME_STS, SCHEME_HTTP, 1)
	ret := struct {
		Length int64 `json:"length"`
	}{}
	if err := lb.Call(ctx, &ret, "POST", uri2); err != nil {
		return uri, 0, err
	}
	if length == nil {
		return uri, ret.Length, nil
	}
	return uri, *length, nil
}

func (lb *lb) Post(ctx context.Context, uri string, length int64, r io.Reader) error {
	uri = strings.Replace(uri, SCHEME_STS, SCHEME_HTTP, 1)
	uri = uri + "?length=" + strconv.FormatInt(length, 10)
	if err := lb.CallWith64(
		ctx, nil,
		"POST", uri, "application/octet-stream",
		r, length,
	); err != nil {
		return nil
	}
	return nil
}

func (lb *lb) SyncPost(
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
	if err := lb.CallWithForm(
		ctx, &id, "POST", uri,
		map[string][]string{"length": []string{strconv.FormatInt(length, 10)}},
	); err != nil {
		syncDone(err)
		xl.Warnf("open sts failed. %s %v", uri, err)
		return err
	}

	syncDone(nil)

	uri = "http://" + host + "/v1/write/" + id.ID
	if err := lb.CallWith64(
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

func (lb *lb) Get(ctx context.Context, uri string, length *int64, options ...GetOption,
) (io.ReadCloser, int64, http.Header, error) {
	req := getRequest{uri: uri, length: length}
	for _, opt := range options {
		opt(&req)
	}
	if req.onlyProxy {
		var options = _OPTION_NONE | OPTION_SYNC
		req.uri, _ = lb.GetURL(ctx, req.uri, req.length, &options)
	} else {
		req.uri, _ = lb.GetURL(ctx, req.uri, req.length, nil)
	}
	req.uri = strings.Replace(req.uri, SCHEME_STS, SCHEME_HTTP, 1)
	req0, err := http.NewRequest("GET", req.uri, nil)
	if err != nil {
		return nil, 0, nil, err
	}
	if req.beginOff != nil {
		HTTP.FormatRangeRequest(req0.Header, req.beginOff, req.endOff)
	}
	resp, err := lb.Do(ctx, req0)
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
