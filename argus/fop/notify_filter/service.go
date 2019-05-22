package notify_filter

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"net/url"
	"strconv"
	"strings"

	"github.com/qiniu/http/httputil.v1"
	"github.com/qiniu/http/restrpc.v1"
	"github.com/qiniu/rpc.v3"
	"github.com/qiniu/xlog.v1"
)

type Service struct {
}

func NewService() Service {
	return Service{}
}

func (s Service) PostHandler(
	ctx context.Context,
	req *struct {
		ReqBody io.ReadCloser
		Cmd     string `json:"cmd"`
		URL     string `json:"url"`
	},
	env *restrpc.Env,
) (interface{}, error) {

	xl, ok := xlog.FromContext(ctx)
	if !ok {
		xl = xlog.New(env.W, env.Req)
		ctx = xlog.NewContext(ctx, xl)
	}

	if mode, _ := base64.StdEncoding.DecodeString(
		env.Req.Header.Get("X-Qiniu-Mode"),
	); string(mode) != "2" { // 只允许异步操作
		xl.Warnf("bad mode. %s", mode)
		return nil, httputil.NewError(http.StatusNotAcceptable, "bad mode")
	}

	if req.Cmd == "" {
		vs, err := url.ParseQuery(env.Req.URL.RawQuery)
		if err != nil {
			return nil, httputil.NewError(http.StatusBadRequest, err.Error())
		}
		req.Cmd = vs.Get("cmd")
		req.URL = vs.Get("url")
	}
	if req.Cmd == "" {
		return nil, httputil.NewError(http.StatusBadRequest, "need params: cmd")
	}

	xl.Infof("req, %+v", req)

	var innerNotifyURL = ""
	var isNotify = false
	var cmdKey = "notify-filter/"
	ops := strings.Split(req.Cmd, "|")
	for _, op := range ops {
		if strings.HasPrefix(op, cmdKey) {
			args := strings.Split(strings.TrimPrefix(op, cmdKey), "/")
			if len(args) > 0 {
				isNotify, _ = strconv.ParseBool(args[0]) // true|false
			}
			if len(args) > 1 {
				innerNotifyURL, _ = url.QueryUnescape(args[1]) // url
			}
		}
	}

	xl.Infof("isNotify=%+v, innerNotifyURL=%s", isNotify, innerNotifyURL)

	var lastRet interface{}
	bs, _ := reqBody(req.URL, req.ReqBody)
	_ = json.Unmarshal(bs, &lastRet)

	if innerNotifyURL != "" {

		uid, bucket, key, err := reqSourceInfo(env.Req)
		if err != nil {
			xl.Errorf("Failed to get SrcInfo, %d, %s, %s, %+v", uid, bucket, key, err)
		} else {
			inReq := struct {
				UID    uint32      `json:"uid"`
				Bucket string      `json:"bucket"`
				Key    string      `json:"key"`
				Result interface{} `json:"result"`
			}{
				UID:    uid,
				Bucket: bucket,
				Key:    key,
				Result: lastRet,
			}
			rpcClient := rpc.Client{Client: &http.Client{}}
			err := rpcClient.CallWithJson(ctx, nil, "POST", innerNotifyURL, inReq)
			xl.Infof("InnerNotify: %+v, %s, %+v", inReq, innerNotifyURL, err)
		}
	}

	if !isNotify {
		// TODO: not notify out, waiting dora pfop param
	}

	env.W.Header().Add("X-Fopd-Multi-Output", "1")
	return lastRet, nil
}

func reqSourceInfo(req *http.Request) (uint32, string, string, error) {
	var (
		uid    uint32
		bucket string
		key    string
	)
	{
		str := req.Header.Get("X-Qiniu-Uid")
		if str == "" {
			return uid, bucket, key, httputil.NewError(http.StatusBadRequest, "need params: uid")
		}
		bs, err := base64.StdEncoding.DecodeString(str)
		if err != nil {
			return uid, bucket, key, httputil.NewError(
				http.StatusBadRequest,
				fmt.Sprintf("bad params: uid, %s", str),
			)
		}
		uid64, err := strconv.ParseUint(string(bs), 10, 64)
		if err != nil {
			return uid, bucket, key, httputil.NewError(
				http.StatusBadRequest,
				fmt.Sprintf("bad params: uid, %s", string(bs)),
			)
		}
		uid = uint32(uid64)
	}

	{
		_bucket := req.Header.Get("X-Qiniu-Bucket")
		if _bucket == "" {
			return uid, bucket, key, httputil.NewError(http.StatusBadRequest, "need bucket param")
		}
		bucketBts, err := base64.StdEncoding.DecodeString(_bucket)
		if err != nil {
			return uid, bucket, key, httputil.NewError(http.StatusBadRequest, "decode bucket name error")
		}
		bucket = string(bucketBts)
	}
	{
		_key := req.Header.Get("X-Qiniu-Key")
		if _key == "" {
			return uid, bucket, key, httputil.NewError(http.StatusBadRequest, "need file key param")
		}
		keyBts, err := base64.StdEncoding.DecodeString(_key)
		if err != nil {
			return uid, bucket, key, httputil.NewError(http.StatusBadRequest, "decode file key error")
		}
		key = string(keyBts)
	}

	return uid, bucket, key, nil
}

func reqBody(url string, reader io.ReadCloser) ([]byte, error) {
	if url == "" {
		bs, _ := ioutil.ReadAll(reader)
		return bs, nil
	}

	resp, err := http.Get(url)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	bs, _ := ioutil.ReadAll(resp.Body)
	return bs, nil
}

//================

type MockReader struct {
	Len int
}

func (m *MockReader) Read(p []byte) (n int, err error) {
	if m.Len <= 0 {
		return 0, io.EOF
	}
	l := m.Len
	if l > len(p) {
		l = len(p)
	}
	m.Len -= l
	return l, nil
}

func (m *MockReader) Close() error {
	return nil
}

type MockService struct {
}

func NewMockService() MockService {
	return MockService{}
}

func (s MockService) PostHandler(
	ctx context.Context,
	req *struct {
		ReqBody io.ReadCloser
		Cmd     string `json:"cmd"`
		URL     string `json:"url"`
	},
	env *restrpc.Env,
) (interface{}, error) {

	var innerNotifyURL = ""
	var isNotify = false
	var cmdKey = "notify-filter/"
	ops := strings.Split(req.Cmd, "|")
	for _, op := range ops {
		if strings.HasPrefix(op, cmdKey) {
			args := strings.Split(strings.TrimPrefix(op, cmdKey), "/")
			if len(args) > 0 {
				isNotify, _ = strconv.ParseBool(args[0]) // true|false
			}
			if len(args) > 1 {
				innerNotifyURL, _ = url.QueryUnescape(args[1]) // url
			}
		}
	}

	_ = isNotify

	if innerNotifyURL == "" {
		return nil, errors.New("no innerNotifyURL")
	}

	return "RESULT", nil
}
