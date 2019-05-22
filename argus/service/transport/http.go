package transport

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/url"
	"reflect"
	"runtime/debug"
	"strconv"
	"strings"
	"time"

	"github.com/go-kit/kit/endpoint"
	httptransport "github.com/go-kit/kit/transport/http"
	log "qiniupkg.com/x/log.v7"

	"github.com/qiniu/http/httputil.v1"
	"github.com/qiniu/rpc.v3"
	xlog "github.com/qiniu/xlog.v1"

	. "qiniu.com/argus/service/service"
)

func MakeHttpServer(end endpoint.Endpoint, reqSample interface{}) *httptransport.Server {
	options := []httptransport.ServerOption{
		httptransport.ServerErrorEncoder(func(ctx context.Context, err error, w http.ResponseWriter) {
			w.Header().Set("X-Reqid", xlog.FromContextSafe(ctx).ReqId())
			info, ok := err.(DetectErrorer)
			if ok {
				httpCode, code, desc := info.DetectError()
				ReplyErr(w, httpCode, code, desc)
			} else {
				code, desc := httputil.DetectError(err)
				httputil.ReplyErr(w, code, desc)
			}
		}),
		httptransport.ServerBefore(func(ctx context.Context, req *http.Request) context.Context {
			return xlog.NewContextWithReq(ctx, req)
		}),
	}

	reqType := reflect.TypeOf(reqSample)
	return httptransport.NewServer(
		func(ctx context.Context, request interface{}) (response interface{}, err error) {
			defer func() {
				if p := recover(); p != nil {
					xl := xlog.FromContextSafe(ctx)
					xl.Errorf("WARN: panic fired - %v\n", p)
					xl.Error(string(debug.Stack()))
					err = ErrInternal(fmt.Sprintf("internal error : %v", p))
				}
			}()
			return end(ctx, request)
		},
		func(_ context.Context, r *http.Request) (request interface{}, err error) {
			var reqValue = reflect.New(reqType)
			if err := json.NewDecoder(r.Body).Decode(reqValue.Interface()); err != nil {
				return nil, err
			}
			return reqValue.Elem().Interface(), nil
		},
		func(ctx context.Context, w http.ResponseWriter, response interface{}) error {
			w.Header().Set("Content-Type", "application/json; charset=utf-8")
			w.Header().Set("X-Reqid", xlog.FromContextSafe(ctx).ReqId())
			return json.NewEncoder(w).Encode(response)
		},
		options...,
	)
}

func MakeHttpClient(method, host string, respSample interface{}) (endpoint.Endpoint, error) {
	if !strings.HasPrefix(host, "http") {
		host = "http://" + host
	}

	_url, err := url.Parse(host)
	if err != nil {
		return endpoint.Nop, err
	}

	options := []httptransport.ClientOption{
		httptransport.SetClient(&http.Client{
			Timeout: 60 * time.Second,
		}),
	}

	respType := reflect.TypeOf(respSample)
	return httptransport.NewClient(method, _url,
			func(ctx context.Context, req *http.Request, request interface{}) error {
				var buf bytes.Buffer
				err := json.NewEncoder(&buf).Encode(request)
				if err != nil {
					return err
				}
				req.Body = ioutil.NopCloser(&buf)
				req.Header.Set("Content-Type", "application/json; charset=utf-8")
				req.Header.Set("Authorization", "QiniuStub uid=1&ut=0")
				req.Header.Set("X-Reqid", xlog.FromContextSafe(ctx).ReqId())
				return nil
			},
			func(ctx context.Context, resp *http.Response) (interface{}, error) {
				if _, ok := respSample.([]byte); ok {
					if resp.StatusCode/100 == 2 {
						defer resp.Body.Close()
						return ioutil.ReadAll(resp.Body)
					}
					return nil, rpc.ResponseError(resp)
				}
				var respValue = reflect.New(respType)
				err := rpc.CallRet(ctx, respValue.Interface(), resp)
				if err != nil {
					return nil, err
				}
				return respValue.Elem().Interface(), nil
			},
			options...).Endpoint(),
		nil
}

func ReplyErr(w http.ResponseWriter, httpCode, code int, detail string) {
	logWithReqid(3, w.Header().Get("X-Reqid"), detail)
	var err struct {
		Code  int    `json:"code"`
		Error string `json:"error"`
	}
	err.Code = code
	err.Error = detail

	msg, _ := json.Marshal(err)
	h := w.Header()
	h.Set("Content-Length", strconv.Itoa(len(msg)))
	h.Set("Content-Type", "application/json")
	w.WriteHeader(httpCode)
	w.Write(msg)
}

func logWithReqid(lvl int, reqid string, str string) {
	str = strings.Replace(str, "\n", "\n["+reqid+"]", -1)
	_ = log.Std.Output(reqid, log.Lwarn, lvl+1, str)
}
