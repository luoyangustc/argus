package rpc

import (
	"bytes"
	"encoding/json"
	"io"
	"io/ioutil"
	"net/http"
	"net/url"
	"strings"
	"time"

	"qiniupkg.com/trace.v1"
)

var UserAgent = "Golang qiniu/rpc package"

// --------------------------------------------------------------------

type Client struct {
	*http.Client
}

var DefaultClient = Client{&http.Client{Transport: DefaultTransport}}

func NewClientTimeout(dial, resp time.Duration) Client {
	return Client{&http.Client{Transport: NewTransportTimeout(dial, resp)}}
}

// --------------------------------------------------------------------

type Logger interface {
	ReqId() string
	Xput(logs []string)
}

// --------------------------------------------------------------------

func (r Client) Head(l Logger, url string) (resp *http.Response, err error) {

	req, err := http.NewRequest("HEAD", url, nil)
	if err != nil {
		return
	}
	return r.Do(l, req)
}

func (r Client) Get(l Logger, url string) (resp *http.Response, err error) {

	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return
	}
	return r.Do(l, req)
}

func (r Client) Delete(l Logger, url string) (resp *http.Response, err error) {

	req, err := http.NewRequest("DELETE", url, nil)
	if err != nil {
		return
	}
	return r.Do(l, req)
}

func (r Client) PostEx(l Logger, url string) (resp *http.Response, err error) {

	req, err := http.NewRequest("POST", url, nil)
	if err != nil {
		return
	}
	return r.Do(l, req)
}

func (r Client) PostWith(
	l Logger, url1 string, bodyType string, body io.Reader, bodyLength int) (resp *http.Response, err error) {

	req, err := http.NewRequest("POST", url1, body)
	if err != nil {
		return
	}
	req.Header.Set("Content-Type", bodyType)
	req.ContentLength = int64(bodyLength)
	return r.Do(l, req)
}

func (r Client) PutWith(
	l Logger, url string, bodyType string, body io.Reader, bodyLength int) (resp *http.Response, err error) {

	req, err := http.NewRequest("PUT", url, body)
	if err != nil {
		return
	}
	req.Header.Set("Content-Type", bodyType)
	req.ContentLength = int64(bodyLength)
	return r.Do(l, req)
}
func (r Client) PostWith64(
	l Logger, url1 string, bodyType string, body io.Reader, bodyLength int64) (resp *http.Response, err error) {

	req, err := http.NewRequest("POST", url1, body)
	if err != nil {
		return
	}
	req.Header.Set("Content-Type", bodyType)
	req.ContentLength = bodyLength
	return r.Do(l, req)
}

func (r Client) PostWithForm(
	l Logger, url1 string, data map[string][]string) (resp *http.Response, err error) {

	msg := url.Values(data).Encode()
	return r.PostWith(l, url1, "application/x-www-form-urlencoded", strings.NewReader(msg), len(msg))
}

func (r Client) PostWithJson(
	l Logger, url1 string, data interface{}) (resp *http.Response, err error) {

	msg, err := json.Marshal(data)
	if err != nil {
		return
	}
	return r.PostWith(l, url1, "application/json", bytes.NewReader(msg), len(msg))
}

func (r Client) PutWithJson(
	l Logger, url1 string, data interface{}) (resp *http.Response, err error) {

	msg, err := json.Marshal(data)
	if err != nil {
		return
	}
	return r.PutWith(l, url1, "application/json", bytes.NewReader(msg), len(msg))
}

func (r Client) Do(l Logger, req *http.Request) (resp *http.Response, err error) {

	t := trace.SafeRecorder(l).Child().Client()
	t.Inject(req)

	e := trace.NewClientEvent(t, req)
	defer func() { t.FlattenKV("http", e.LogResponse(resp, err)).Finish() }()

	if l != nil {
		t.Tag("reqid", l.ReqId())
		req.Header.Set("X-Reqid", l.ReqId())
	}
	if req.Header.Get("User-Agent") == "" {
		req.Header.Set("User-Agent", UserAgent)
	}
	resp, err = r.Client.Do(req)
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

type ErrorInfo struct {
	Err     string   `json:"error"`
	Reqid   string   `json:"reqid"`
	Details []string `json:"details"`
	Code    int      `json:"code"`
}

func (r *ErrorInfo) ErrorDetail() string {
	msg, _ := json.Marshal(r)
	return string(msg)
}

func (r *ErrorInfo) Error() string {
	if r.Err != "" {
		return r.Err
	}
	return http.StatusText(r.Code)
}

func (r *ErrorInfo) HttpCode() int {
	return r.Code
}

// --------------------------------------------------------------------

type errorRet struct {
	Error string `json:"error"`
}

func ResponseError(resp *http.Response) (err error) {

	e := &ErrorInfo{
		Details: resp.Header["X-Log"],
		Reqid:   resp.Header.Get("X-Reqid"),
		Code:    resp.StatusCode,
	}
	if resp.StatusCode > 299 {
		if resp.ContentLength != 0 {
			if ct := resp.Header.Get("Content-Type"); strings.TrimSpace(strings.SplitN(ct, ";", 2)[0]) == "application/json" {
				var ret1 errorRet
				json.NewDecoder(resp.Body).Decode(&ret1)
				e.Err = ret1.Error
			}
		}
	}
	return e
}

func CallRet(l Logger, ret interface{}, resp *http.Response) (err error) {

	return callRet(l, ret, resp)
}

func callRet(l Logger, ret interface{}, resp *http.Response) (err error) {

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
		if resp.StatusCode == 200 || resp.StatusCode == 204 {
			return nil
		}
	}
	return ResponseError(resp)
}

func (r Client) CallWithForm(l Logger, ret interface{}, url1 string, param map[string][]string) (err error) {

	resp, err := r.PostWithForm(l, url1, param)
	if err != nil {
		return err
	}
	return callRet(l, ret, resp)
}

func (r Client) CallWithJson(l Logger, ret interface{}, url1 string, param interface{}) (err error) {

	resp, err := r.PostWithJson(l, url1, param)
	if err != nil {
		return err
	}
	return callRet(l, ret, resp)
}

func (r Client) PutCallWithJson(l Logger, ret interface{}, url1 string, param interface{}) (err error) {

	resp, err := r.PutWithJson(l, url1, param)
	if err != nil {
		return err
	}

	return callRet(l, ret, resp)
}

func (r Client) CallWith(
	l Logger, ret interface{}, url1 string, bodyType string, body io.Reader, bodyLength int) (err error) {

	resp, err := r.PostWith(l, url1, bodyType, body, bodyLength)
	if err != nil {
		return err
	}
	return callRet(l, ret, resp)
}

func (r Client) CallWith64(
	l Logger, ret interface{}, url1 string, bodyType string, body io.Reader, bodyLength int64) (err error) {

	resp, err := r.PostWith64(l, url1, bodyType, body, bodyLength)
	if err != nil {
		return err
	}
	return callRet(l, ret, resp)
}

func (r Client) Call(
	l Logger, ret interface{}, url1 string) (err error) {

	resp, err := r.PostWith(l, url1, "application/x-www-form-urlencoded", nil, 0)
	if err != nil {
		return err
	}
	return callRet(l, ret, resp)
}

func (r Client) GetCall(l Logger, ret interface{}, url1 string) (err error) {
	resp, err := r.Get(l, url1)
	if err != nil {
		return err
	}
	return callRet(l, ret, resp)
}

func (r Client) GetCallWithForm(l Logger, ret interface{}, url1 string, param map[string][]string) (err error) {
	payload := url.Values(param).Encode()

	if strings.ContainsRune(url1, '?') {
		url1 += "&"
	} else {
		url1 += "?"
	}
	url1 += payload

	resp, err := r.Get(l, url1)
	if err != nil {
		return err
	}
	return callRet(l, ret, resp)
}

func (r Client) DeleteCall(l Logger, ret interface{}, url string) (err error) {
	resp, err := r.Delete(l, url)
	if err != nil {
		return err
	}
	return callRet(l, ret, resp)
}

// --------------------------------------------------------------------
