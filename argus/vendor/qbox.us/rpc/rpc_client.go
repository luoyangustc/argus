package rpc

import (
	"encoding/json"
	"io"
	"io/ioutil"
	"net/http"
	"net/url"
	"qbox.us/api"
	"qbox.us/cc"
	"qbox.us/errors"
	"qbox.us/multipart"
	"strings"
)

// --------------------------------------------------------------------
// Client

//
// TODO: 此类可能被废除，建议使用 qbox.us/net/httputil.Client 类。
//
type Client struct {
	*http.Client
}

func (r Client) PostEx(url_ string, bodyType string, body io.Reader, bodyLength int) (resp *http.Response, err error) {
	req, err := http.NewRequest("POST", url_, body)
	if err != nil {
		return
	}
	req.Header.Set("Content-Type", bodyType)
	req.ContentLength = int64(bodyLength)
	return r.Do(req)
}

func (r Client) PostEx64(url_ string, bodyType string, body io.Reader, bodyLength int64) (resp *http.Response, err error) {
	req, err := http.NewRequest("POST", url_, body)
	if err != nil {
		return
	}
	req.Header.Set("Content-Type", bodyType)
	req.ContentLength = bodyLength
	return r.Do(req)
}

func (r Client) PostMultipart(url_ string, data map[string][]string) (resp *http.Response, err error) {
	body, ct, err := multipart.Open(data)
	if err != nil {
		return
	}
	defer body.Close()

	return r.PostEx(url_, ct, body, -1)
}

func (r Client) PostForm(url_ string, data map[string][]string) (resp *http.Response, err error) {
	msg := url.Values(data).Encode()
	return r.PostEx(url_, "application/x-www-form-urlencoded", strings.NewReader(msg), len(msg))
}

func (r Client) PostNoParam(url_ string) (resp *http.Response, err error) {
	return r.PostEx(url_, "application/x-www-form-urlencoded", nil, 0)
}

func (r Client) PostWithParam(url_ string, bodyType string, msg []byte) (resp *http.Response, err error) {
	body := cc.NewBytesReader(msg)
	return r.PostEx(url_, bodyType, body, len(msg))
}

func (r Client) PostBinary(url_ string, body io.Reader, bodyLength int) (resp *http.Response, err error) {
	return r.PostEx(url_, "application/octet-stream", body, bodyLength)
}

func (r Client) PostBinary64(url_ string, body io.Reader, bodyLength int64) (resp *http.Response, err error) {
	return r.PostEx64(url_, "application/octet-stream", body, bodyLength)
}

func (r Client) PostJson(url_ string, param interface{}) (resp *http.Response, err error) {
	msg, _ := json.Marshal(param)
	body := cc.NewBytesReader(msg)
	return r.PostEx(url_, "application/json", body, len(msg))
}

func callRet(ret interface{}, resp *http.Response) (code int, err error) {

	defer func() {
		io.Copy(ioutil.Discard, resp.Body)
		resp.Body.Close()
	}()

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
		if resp.ContentLength != 0 {
			if ct, ok := resp.Header["Content-Type"]; ok && ct[0] == "application/json" {
				var ret1 ErrorRet
				json.NewDecoder(resp.Body).Decode(&ret1)
				if ret1.Error != "" {
					err = errors.Info(errors.New(ret1.Error), "jsonrpc.callRet")
					return
				}
			}
		}
		err = errors.Info(api.NewError(code), "jsonrpc.callRet")
	}
	return
}

func (r Client) CallWithParam(ret interface{}, url_ string, bodyType string, msg []byte) (code int, err error) {
	body := cc.NewBytesReader(msg)
	resp, err := r.PostEx(url_, bodyType, body, len(msg))
	if err != nil {
		return api.NetworkError, err
	}
	return callRet(ret, resp)
}

func (r Client) CallWithJson(ret interface{}, url_ string, param interface{}) (code int, err error) {
	resp, err := r.PostJson(url_, param)
	if err != nil {
		return api.NetworkError, err
	}
	return callRet(ret, resp)
}

func (r Client) CallWithForm(ret interface{}, url_ string, param map[string][]string) (code int, err error) {
	resp, err := r.PostForm(url_, param)
	if err != nil {
		return api.NetworkError, err
	}
	return callRet(ret, resp)
}

func (r Client) CallWithMultipart(ret interface{}, url_ string, param map[string][]string) (code int, err error) {
	resp, err := r.PostMultipart(url_, param)
	if err != nil {
		return api.NetworkError, err
	}
	return callRet(ret, resp)
}

func (r Client) CallWithBinary1(ret interface{}, url_ string, body io.Reader) (code int, err error) {
	resp, err := r.Post(url_, "application/octet-stream", body)
	if err != nil {
		return api.NetworkError, err
	}
	return callRet(ret, resp)
}

func (r Client) CallWithBinary(ret interface{}, url_ string, body io.Reader, bodyLength int) (code int, err error) {
	resp, err := r.PostEx(url_, "application/octet-stream", body, bodyLength)
	if err != nil {
		return api.NetworkError, err
	}
	return callRet(ret, resp)
}

func (r Client) CallWithBinary64(ret interface{}, url_ string, body io.Reader, bodyLength int64) (code int, err error) {
	resp, err := r.PostEx64(url_, "application/octet-stream", body, bodyLength)
	if err != nil {
		return api.NetworkError, err
	}
	return callRet(ret, resp)
}

func (r Client) CallWithBinaryEx(
	ret interface{}, url_ string, bodyType string, body io.Reader, bodyLength int) (code int, err error) {

	resp, err := r.PostEx(url_, bodyType, body, bodyLength)
	if err != nil {
		return api.NetworkError, err
	}
	return callRet(ret, resp)
}

func (r Client) CallWithBinaryEx64(
	ret interface{}, url_ string, bodyType string, body io.Reader, bodyLength int64) (code int, err error) {

	resp, err := r.PostEx64(url_, bodyType, body, bodyLength)
	if err != nil {
		return api.NetworkError, err
	}
	return callRet(ret, resp)
}

func (r Client) Call(ret interface{}, url_ string) (code int, err error) {
	resp, err := r.PostEx(url_, "application/x-www-form-urlencoded", nil, 0)
	if err != nil {
		return api.NetworkError, err
	}
	return callRet(ret, resp)
}

// --------------------------------------------------------------------
// Helper

var DefaultClient = Client{http.DefaultClient}

func CallWithBinary(ret interface{}, url_ string, body io.Reader, bodyLength int) (code int, err error) {
	return DefaultClient.CallWithBinary(ret, url_, body, bodyLength)
}

func CallWithBinary64(ret interface{}, url_ string, body io.Reader, bodyLength int64) (code int, err error) {
	return DefaultClient.CallWithBinary64(ret, url_, body, bodyLength)
}

func CallWithMultipart(ret interface{}, url_ string, param map[string][]string) (code int, err error) {
	return DefaultClient.CallWithMultipart(ret, url_, param)
}

func CallWithForm(ret interface{}, url_ string, param map[string][]string) (code int, err error) {
	return DefaultClient.CallWithForm(ret, url_, param)
}

func CallWithJson(ret interface{}, url_ string, param interface{}) (code int, err error) {
	return DefaultClient.CallWithJson(ret, url_, param)
}

func Call(ret interface{}, url_ string) (code int, err error) {
	return DefaultClient.Call(ret, url_)
}
