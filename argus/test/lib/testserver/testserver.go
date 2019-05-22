package main

import (
	"encoding/json"
	"flag"
	"io/ioutil"
	"net/http"
	"net/url"
	"time"

	"strconv"

	"github.com/qiniu/http/httputil.v1"
	"github.com/qiniu/log.v1"
	"qbox.us/api"
)

// ---------------------------------------------------------------------------
// type ErrorInfo

type ErrorInfo struct {
	Code int
	Err  string
}

var ErrHTTPRequest = ErrorInfo{400, "Wrong url, please use /slowResponse?sleep=<time>"}

func handler(w http.ResponseWriter, r *http.Request) {
	log.Info(r.URL.String())

	bs, _ := ioutil.ReadAll(r.Body)
	if r.Header.Get("Content-Type") == "application/json" {
		w.WriteHeader(200)
		w.Write(bs)
		return
	}

	m, err := url.ParseQuery(string(bs))
	if err != nil {
		err = httputil.NewError(api.InvalidArgs, "invalid query body")
		httputil.Error(w, err)
		return
	}

	ret := make(map[string]string)
	for k, vs := range m {
		ret[k] = vs[0]
	}
	httputil.Reply(w, 200, ret)
}

func handlerFetchKey(w http.ResponseWriter, r *http.Request) {
	log.Info(r.URL.String())

	if r.Header.Get("Content-Type") == "application/json" {
		var payload interface{}
		dec := json.NewDecoder(r.Body)
		dec.UseNumber()
		_ = dec.Decode(&payload)
		key := time.Now().String()
		ret := map[string]interface{}{
			"key":     key,
			"payload": payload,
		}
		httputil.Reply(w, 200, ret)
		return
	}

	bs, _ := ioutil.ReadAll(r.Body)
	m, err := url.ParseQuery(string(bs))
	if err != nil {
		err = httputil.NewError(api.InvalidArgs, "invalid query body")
		httputil.Error(w, err)
		return
	}

	if ua := r.Header.Get("User-Agent"); ua != "qiniu-callback/1.0" {
		err = httputil.NewError(500, "unknown ua: "+ua)
		httputil.Error(w, err)
		return
	}

	payload := make(map[string]string)
	for k, vs := range m {
		payload[k] = vs[0]
	}
	key := time.Now().String()
	ret := map[string]interface{}{
		"key":     key,
		"payload": payload,
	}
	httputil.Reply(w, 200, ret)
}

//此接口构造一个慢响应的请求
//使用方法:
// 1. /slowResponse
//    默认是等待30秒之后返回
// 2. /slowResponse?sleep=<期望等待多少秒>
//    传入期望等待的秒数
//
func handlerSlowResponse(w http.ResponseWriter, r *http.Request) {
	log.Info(r.URL.String())

	value := r.URL.Query()
	defaultSleep := 30 * time.Second

	if value == nil || len(value) == 0 {
		time.Sleep(defaultSleep)
	} else {
		n, err := strconv.Atoi(value["sleep"][0])

		if err != nil {
			httputil.Reply(w, ErrHTTPRequest.Code, ErrHTTPRequest.Err)
			return
		}
		time.Sleep(time.Duration(int64(n)) * time.Second)
	}

	w.WriteHeader(200)
	w.Write([]byte("ok"))
}

/*
 此接口主要测试Issue: https://pm.qbox.me/issues/23349
 实现两个功能：
 1. 直接返回response, 不含 Content-Length头
 2. 如果request 中包含Accept-Encoding, 那么返回带Content-Encoding的Response
*/
func handlerCustomHeaders(rw http.ResponseWriter, req *http.Request) {
	log.Info("handlerCustomHeaders: " + req.URL.String())

	if req.Header.Get("Accept-Encoding") != "" {
		log.Info("Request contains Accept-Encoding header")
		rw.Header().Set("Content-Encoding", "Fill")
	}
	rw.Header().Set("Transfer-Encoding", "chunked")
	rw.WriteHeader(http.StatusOK)
	rw.Write([]byte("OK"))
}

//
// 测试 issue: https://pm.qbox.me/issues/26385
// 提供接口: /pfop
//          sleep 20 秒之后返回
func mockPfopHost(rw http.ResponseWriter, req *http.Request) {
	log.Info("mockPfopHost: " + req.URL.String())
	time.Sleep(time.Duration(20) * time.Second)

	rw.WriteHeader(200)
	rw.Write([]byte("ok"))
}

//此接口构造一个服务,返回你期望的 HTTP Code
//使用方法:
// 1. /customeResponseCode?code=<期望返回的HTTP Code值>&sleep=<期望多少Millisecond后返回结果>
//
func customeResponseCode(w http.ResponseWriter, r *http.Request) {
	log.Info(r.URL.String())

	value := r.URL.Query()
	code := 200
	if value != nil {
		c, err := strconv.Atoi(value["code"][0])
		if err != nil {
			httputil.Reply(w, ErrHTTPRequest.Code, ErrHTTPRequest.Err)
			return
		}
		code = c

		ss := value["sleep"]
		if ss != nil {
			s, e := strconv.Atoi(ss[0])
			if e != nil {
				log.Fatal("Unable to parse sleep parameter", e)
			}

			time.Sleep(time.Duration(s) * time.Millisecond)
		}
	}

	w.WriteHeader(code)
	w.Write([]byte("HELLO"))
}

func main() {
	var addr string
	flag.StringVar(&addr, "addr", ":8090", "")
	flag.Parse()

	http.HandleFunc("/callback", handler)
	http.HandleFunc("/callbackFetchKey", handlerFetchKey)
	http.HandleFunc("/slowResponse", handlerSlowResponse)
	http.HandleFunc("/customHeaders", handlerCustomHeaders)
	http.HandleFunc("/pfop", mockPfopHost)
	http.HandleFunc("/customeResponseCode", customeResponseCode)

	log.Info("listen on", addr)
	log.Fatalln(http.ListenAndServe(addr, nil))
}
