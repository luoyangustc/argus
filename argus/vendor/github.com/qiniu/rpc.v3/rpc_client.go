package rpc

import (
	"bytes"
	. "context" // 手动修改，与Base库不符合
	"encoding/json"
	"io"
	"io/ioutil"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/qiniu/http/httputil.v1"
	"github.com/qiniu/xlog.v1"

	"qiniupkg.com/trace.v1"
)

var UserAgent = "Golang qiniu/rpc package"

// --------------------------------------------------------------------

var DefaultClient = NewClientWithTransport(http.DefaultTransport)

type Client struct {
	*http.Client
}

func NewClientWithTransport(tp http.RoundTripper) Client {

	if tp == nil {
		tp = http.DefaultTransport
	}
	return Client{&http.Client{Transport: tp}}
}

// --------------------------------------------------------------------

func (r Client) DoRequest(ctx Context, method, url string) (resp *http.Response, err error) {

	req, err := http.NewRequest(method, url, nil)
	if err != nil {
		return
	}
	return r.Do(ctx, req)
}

func (r Client) DoRequestWith(
	ctx Context, method, url1 string,
	bodyType string, body io.Reader, bodyLength int) (resp *http.Response, err error) {

	req, err := http.NewRequest(method, url1, body)
	if err != nil {
		return
	}
	req.Header.Set("Content-Type", bodyType)
	req.ContentLength = int64(bodyLength)
	return r.Do(ctx, req)
}

func (r Client) DoRequestWith64(
	ctx Context, method, url1 string,
	bodyType string, body io.Reader, bodyLength int64) (resp *http.Response, err error) {

	req, err := http.NewRequest(method, url1, body)
	if err != nil {
		return
	}
	req.Header.Set("Content-Type", bodyType)
	req.ContentLength = bodyLength
	return r.Do(ctx, req)
}

func (r Client) DoRequestWith64Header(
	ctx Context, method, url1 string,
	bodyType string, body io.Reader, bodyLength int64, m map[string]string) (resp *http.Response, err error) {

	req, err := http.NewRequest(method, url1, body)
	if err != nil {
		return
	}
	req.Header.Set("Content-Type", bodyType)
	for key, value := range m {
		req.Header.Set(key, value)
	}
	req.ContentLength = bodyLength
	return r.Do(ctx, req)
}

func (r Client) DoRequestWithForm(
	ctx Context, method, url1 string, data map[string][]string) (resp *http.Response, err error) {

	msg := url.Values(data).Encode()
	if method == "GET" || method == "HEAD" || method == "DELETE" {
		if strings.ContainsRune(url1, '?') {
			url1 += "&"
		} else {
			url1 += "?"
		}
		return r.DoRequest(ctx, method, url1+msg)
	}
	return r.DoRequestWith(
		ctx, method, url1, "application/x-www-form-urlencoded", strings.NewReader(msg), len(msg))
}

func (r Client) DoRequestWithJson(
	ctx Context, method, url1 string, data interface{}) (resp *http.Response, err error) {

	msg, err := json.Marshal(data)
	if err != nil {
		return
	}
	return r.DoRequestWith(
		ctx, method, url1, "application/json", bytes.NewReader(msg), len(msg))
}

func (r Client) Do(ctx Context, req *http.Request) (resp *http.Response, err error) {

	if ctx == nil {
		ctx = Background()
	}
	l := xlog.FromContextSafe(ctx)
	req.Header.Set("X-Reqid", l.ReqId())
	if req.Header.Get("User-Agent") == "" {
		req.Header.Set("User-Agent", UserAgent)
	}

	transport := r.Transport // don't change r.Transport
	if transport == nil {
		transport = http.DefaultTransport
	}

	// avoid cancel() is called before Do(req), but isn't accurate
	select {
	case <-ctx.Done():
		err = ctx.Err()
		return
	default:
	}

	t := l.T().Child().Tag("reqid", l.ReqId()).Client()
	t.Inject(req)

	e := trace.NewClientEvent(t, req)
	defer func() { t.FlattenKV("http", e).Finish() }()

	// TODO:
	// 假如服务端的response已经返回了
	// 这时候开始读数据，然后Context被cancel了，这个库不会处理
	if tr, ok := httputil.GetRequestCanceler(transport); ok { // support CancelRequest
		reqC := make(chan bool, 1)
		start := time.Now()
		go func() {
			n := time.Now()
			if n.Add(-time.Second).After(start) {
				l.Errorf("start at %v, do at %v", start, n)
			}
			resp, err = r.Client.Do(req)
			e.LogResponse(resp, err)
			reqC <- true
		}()
		select {
		case <-reqC:
		case <-ctx.Done():
			tr.CancelRequest(req)
			<-reqC
			if err == nil {
				// 有可能r.Client.Do还没有调用到就进入这个逻辑
				// 需要关掉这个http请求，否则会造成tcp连接泄露
				resp.Body.Close()
			}
			err = ctx.Err()
		}
	} else {
		resp, err = r.Client.Do(req)
		e.LogResponse(resp, err)
	}
	if err != nil {
		return
	}

	if l != nil {
		details := resp.Header["X-Log"]
		if len(details) > 0 {
			l.Xput(details)
		}
	}
	return
}

// --------------------------------------------------------------------

type RespError interface {
	ErrorDetail() string
	Error() string
	HttpCode() int
}

type errorInfo struct {
	Err     string   `json:"error"`
	Reqid   string   `json:"reqid"`
	Details []string `json:"details"`
	Code    int      `json:"code"`
}

func (r *errorInfo) ErrorDetail() string {
	msg, _ := json.Marshal(r)
	return string(msg)
}

func (r *errorInfo) Error() string {
	if r.Err != "" {
		return r.Err
	}
	return http.StatusText(r.Code)
}

func (r *errorInfo) HttpCode() int {
	return r.Code
}

// --------------------------------------------------

type httpCoder interface {
	HttpCode() int
}

func HttpCodeOf(err error) int {
	if hc, ok := err.(httpCoder); ok {
		return hc.HttpCode()
	}
	return 0
}

// --------------------------------------------------------------------

type errorRet struct {
	Error string `json:"error"`
}

func parseError(r io.Reader) (err string) {

	body, err1 := ioutil.ReadAll(r)
	if err1 != nil {
		return err1.Error()
	}

	m := make(map[string]interface{})
	json.Unmarshal(body, &m)
	if e, ok := m["error"]; ok {
		if err, ok = e.(string); ok {
			// qiniu error msg style returns here
			return
		}
	}
	return string(body)
}

func ResponseError(resp *http.Response) (err error) {

	e := &errorInfo{
		Details: resp.Header["X-Log"],
		Reqid:   resp.Header.Get("X-Reqid"),
		Code:    resp.StatusCode,
	}
	if resp.StatusCode > 299 {
		if resp.ContentLength != 0 {
			if ct := resp.Header.Get("Content-Type"); strings.TrimSpace(strings.SplitN(ct, ";", 2)[0]) == "application/json" {
				e.Err = parseError(resp.Body)
			}
		}
	}
	return e
}

func CallRet(ctx Context, ret interface{}, resp *http.Response) (err error) {

	defer func() {
		io.Copy(ioutil.Discard, resp.Body)
		resp.Body.Close()
	}()

	if resp.StatusCode/100 == 2 {
		if ret != nil && resp.ContentLength != 0 {
			err = json.NewDecoder(resp.Body).Decode(ret)
			if err != nil {
				return
			}
		}
		if resp.StatusCode == 200 {
			return nil
		}
	}
	return ResponseError(resp)
}

func (r Client) CallWithForm(
	ctx Context, ret interface{}, method, url1 string, param map[string][]string) (err error) {

	resp, err := r.DoRequestWithForm(ctx, method, url1, param)
	if err != nil {
		return err
	}
	return CallRet(ctx, ret, resp)
}

func (r Client) CallWithJson(
	ctx Context, ret interface{}, method, url1 string, param interface{}) (err error) {

	resp, err := r.DoRequestWithJson(ctx, method, url1, param)
	if err != nil {
		return err
	}
	return CallRet(ctx, ret, resp)
}

func (r Client) CallWith(
	ctx Context, ret interface{}, method, url1, bodyType string, body io.Reader, bodyLength int) (err error) {

	resp, err := r.DoRequestWith(ctx, method, url1, bodyType, body, bodyLength)
	if err != nil {
		return err
	}
	return CallRet(ctx, ret, resp)
}

func (r Client) CallWith64(
	ctx Context, ret interface{}, method, url1, bodyType string, body io.Reader, bodyLength int64) (err error) {

	resp, err := r.DoRequestWith64(ctx, method, url1, bodyType, body, bodyLength)
	if err != nil {
		return err
	}
	return CallRet(ctx, ret, resp)
}

func (r Client) Call(
	ctx Context, ret interface{}, method, url1 string) (err error) {

	var resp *http.Response
	if method == "HEAD" || method == "GET" {
		resp, err = r.DoRequest(ctx, method, url1)
		if err != nil {
			return err
		}
	} else {
		resp, err = r.DoRequestWith(ctx, method, url1, "application/x-www-form-urlencoded", nil, 0)
		if err != nil {
			return err
		}
	}
	return CallRet(ctx, ret, resp)
}

// --------------------------------------------------------------------
