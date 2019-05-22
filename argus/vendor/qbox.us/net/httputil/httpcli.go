package httputil

import (
	"encoding/json"
	"io"
	"net/http"
	"net/url"
	"strings"

	"qbox.us/api"
	"qbox.us/errors"

	"github.com/qiniu/http/httputil.v1"
)

// --------------------------------------------------------------------

type Client struct {
	*http.Client
}

// --------------------------------------------------------------------

//过时函数，请调用 qiniu/rpc 包对应函数
func (r Client) PostWith(url1 string, bodyType string, body io.Reader, bodyLength int) (resp *http.Response, err error) {
	req, err := http.NewRequest("POST", url1, body)
	if err != nil {
		return
	}
	req.Header.Set("Content-Type", bodyType)
	req.ContentLength = int64(bodyLength)
	return r.Do(req)
}

//过时函数，请调用 qiniu/rpc 包对应函数
func (r Client) PostWith64(url1 string, bodyType string, body io.Reader, bodyLength int64) (resp *http.Response, err error) {
	req, err := http.NewRequest("POST", url1, body)
	if err != nil {
		return
	}
	req.Header.Set("Content-Type", bodyType)
	req.ContentLength = bodyLength
	return r.Do(req)
}

//过时函数，请调用 qiniu/rpc 包对应函数
func (r Client) PostWithForm(url1 string, data map[string][]string) (resp *http.Response, err error) {
	msg := url.Values(data).Encode()
	return r.PostWith(url1, "application/x-www-form-urlencoded", strings.NewReader(msg), len(msg))
}

// --------------------------------------------------------------------

type ErrorRet struct {
	Error string "error"
}

func ResponseError(resp *http.Response) (err error) {

	if resp.ContentLength != 0 {
		if ct := resp.Header.Get("Content-Type"); strings.TrimSpace(strings.SplitN(ct, ";", 2)[0]) == "application/json" {
			var ret1 ErrorRet
			json.NewDecoder(resp.Body).Decode(&ret1)
			if ret1.Error != "" {
				err = httputil.NewError(resp.StatusCode, ret1.Error)
				err = errors.Info(err, "jsonrpc.callRet")
				return
			}
		}
	}
	err = errors.Info(api.NewError(resp.StatusCode), "jsonrpc.callRet")
	return
}

func callRet(ret interface{}, resp *http.Response) (code int, err error) {

	defer resp.Body.Close()

	code = resp.StatusCode
	if code/100 == 2 {
		if ret != nil && resp.ContentLength != 0 {
			err = json.NewDecoder(resp.Body).Decode(ret)
			if err != nil {
				err = errors.Info(api.EUnexpectedResponse, "jsonrpc.callRet").Detail(err)
				code = api.UnexpectedResponse
			}
		}
	} else {
		err = ResponseError(resp)
	}
	return
}

//过时函数，请调用 qiniu/rpc 包对应函数
func (r Client) CallWithForm(ret interface{}, url1 string, param map[string][]string) (code int, err error) {
	resp, err := r.PostWithForm(url1, param)
	if err != nil {
		return api.NetworkError, err
	}
	return callRet(ret, resp)
}

//过时函数，请调用 qiniu/rpc 包对应函数
func (r Client) CallWith(ret interface{}, url1 string, bodyType string, body io.Reader, bodyLength int) (code int, err error) {

	resp, err := r.PostWith(url1, bodyType, body, bodyLength)
	if err != nil {
		return api.NetworkError, err
	}
	return callRet(ret, resp)
}

//过时函数，请调用 qiniu/rpc 包对应函数
func (r Client) CallWith64(ret interface{}, url1 string, bodyType string, body io.Reader, bodyLength int64) (code int, err error) {

	resp, err := r.PostWith64(url1, bodyType, body, bodyLength)
	if err != nil {
		return api.NetworkError, err
	}
	return callRet(ret, resp)
}

//过时函数，请调用 qiniu/rpc 包对应函数
func (r Client) Call(ret interface{}, url1 string) (code int, err error) {
	resp, err := r.PostWith(url1, "application/x-www-form-urlencoded", nil, 0)
	if err != nil {
		return api.NetworkError, err
	}
	return callRet(ret, resp)
}

// --------------------------------------------------------------------
