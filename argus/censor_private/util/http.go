package util

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"strings"
	"time"

	"github.com/qiniu/rpc.v3"
)

type ErrorInfo struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

func (err *ErrorInfo) Error() string {
	return err.Message
}

func GetWithContent(ctx context.Context, timeout time.Duration, url string) ([]byte, error) {
	client := newClient(timeout)
	resp, err := client.DoRequest(ctx, "GET", url)
	if err != nil {
		return nil, err
	}

	defer resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		return nil, rpc.ResponseError(resp)
	}

	return ioutil.ReadAll(resp.Body)
}

func PostJson(ctx context.Context, timeout time.Duration,
	url string, req interface{}, ret interface{}) error {
	// 服务返回的错误将用标准七牛的error {"error": "xxx"}
	client := newClient(timeout)
	return client.CallWithJson(ctx, ret, "POST", url, req)
}

func PostJsonWithCensorError(ctx context.Context, timeout time.Duration,
	url string, req interface{}, ret interface{}) error {
	// 服务返回的错误将用审核的error {"code": xxx, "error": "xxx"}
	client := newClient(timeout)

	resp, err := client.DoRequestWithJson(ctx, "POST", url, req)
	if err != nil {
		return err
	}
	return callRet(ctx, ret, resp)
}

func newClient(timeout time.Duration) *rpc.Client {
	return &rpc.Client{
		Client: &http.Client{
			Timeout: timeout,
		},
	}
}

func callRet(ctx context.Context, ret interface{}, resp *http.Response) (err error) {

	defer func() {
		_, _ = io.Copy(ioutil.Discard, resp.Body)
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
	return responseError(resp)
}

func responseError(resp *http.Response) (err error) {

	httpcode := resp.StatusCode
	var code int
	var msg string
	if httpcode > 299 {
		if resp.ContentLength != 0 {
			if ct := resp.Header.Get("Content-Type"); strings.TrimSpace(strings.SplitN(ct, ";", 2)[0]) == "application/json" {
				body, err1 := ioutil.ReadAll(resp.Body)
				if err1 != nil {
					return err1
				}

				m := make(map[string]interface{})
				_ = json.Unmarshal(body, &m)
				if v, ok := m["error"]; ok {
					msg, _ = v.(string)
				}
				if v, ok := m["code"]; ok {
					if vv, ok := v.(float64); ok {
						code = int(vv)
					}
				}
			}
		}
	}

	if len(msg) == 0 {
		return fmt.Errorf(http.StatusText(httpcode))
	}

	if code != 0 {
		return &ErrorInfo{
			Code:    code,
			Message: msg,
		}
	}

	return fmt.Errorf(msg)
}
