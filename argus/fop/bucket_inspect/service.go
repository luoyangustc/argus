package bucket_inspect

import (
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	qboxmac "qiniu.com/auth/qboxmac.v1"

	"qbox.us/api/one/access"
	"qbox.us/qconf/qconfapi"

	"github.com/qiniu/http/httputil.v1"
	"github.com/qiniu/http/restrpc.v1"
	"github.com/qiniu/rpc.v3"
	"github.com/qiniu/xlog.v1"
)

type Config struct {
	RsHost string          `json:"rs_host"`
	Qconf  qconfapi.Config `json:"qconf"`
	// AppdAccessHost string          `json:"appd_access_host"`
}

type Service struct {
	Config
	qCLI *qconfapi.Client
}

func NewService(conf Config) Service {
	return Service{conf, qconfapi.New(&conf.Qconf)}
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

	var (
		ret     interface{}
		disable = false
		err     error
	)

	switch lastCmd, args := func() (string, []string) {

		bs, _ := base64.StdEncoding.DecodeString(env.Req.Header.Get("X-Qiniu-Pipe-Cmds"))
		xl.Infof("cmds: %s", string(bs))
		var (
			lastCmd string
			args    []string
		)
		ops := strings.Split(string(bs), "|")
		for i, op := range ops {
			if strings.HasPrefix(op, "bucket-inspect/") {
				if i > 0 {
					strs := strings.SplitN(ops[i-1], "/", 2)
					if len(strs) >= 1 {
						lastCmd = strs[0]
					}
					args = strings.Split(strings.TrimPrefix(op, "bucket-inspect/"), "/")
				}
				break
			}
		}
		return lastCmd, args

	}(); lastCmd {
	case "qpulp":
		ret, disable, err = qpulp(ctx, args, req)
		if err != nil {
			return nil, err
		}
	case "image-censor":
		ret, disable, err = imageCensor(ctx, args, req)
		if err != nil {
			return nil, err
		}
	case "video-censor":
		ret, disable, err = videoCensor(ctx, args, req)
		if err != nil {
			return nil, err
		}
	default:
		xl.Warnf("bad last cmd. %s", lastCmd)
		return nil, httputil.NewError(http.StatusBadRequest, "bad ops")
	}

	if disable {
		if err := s.disable(ctx, env.Req); err != nil {
			xl.Errorf("disable failed. %v", err)
		}
	}

	env.W.Header().Add("X-Fopd-Multi-Output", "1")
	return struct {
		Result struct {
			Result  interface{} `json:"result"`
			Disable bool        `json:"disable"`
		} `json:"result"`
	}{
		Result: struct {
			Result  interface{} `json:"result"`
			Disable bool        `json:"disable"`
		}{Result: ret, Disable: disable},
	}, nil
}

func aksk2(qCLI *qconfapi.Client, uid uint32) (string, string, error) {

	if qCLI == nil {
		return "", "", errors.New("qCLI nil")
	}

	var ret access.AppInfo
	err := qCLI.Get(
		xlog.NewDummy(), &ret,
		"app:"+strconv.FormatUint(uint64(uid), 36)+":default",
		0)
	if err != nil {
		return "", "", err
	}
	return ret.Key, ret.Secret, nil
}

func reqUID(req *http.Request) (uint32, error) {
	var uid uint64
	str := req.Header.Get("X-Qiniu-Uid")
	if str == "" {
		return 0, httputil.NewError(http.StatusBadRequest, "need params: uid")
	}
	bs, err := base64.StdEncoding.DecodeString(str)
	if err != nil {
		return 0, httputil.NewError(
			http.StatusBadRequest,
			fmt.Sprintf("bad params: uid, %s", str),
		)
	}
	uid, err = strconv.ParseUint(string(bs), 10, 64)
	if err != nil {
		return 0, httputil.NewError(
			http.StatusBadRequest,
			fmt.Sprintf("bad params: uid, %s", string(bs)),
		)
	}
	return uint32(uid), nil
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

func (s Service) disable(ctx context.Context, req *http.Request) error {

	if req == nil {
		return errors.New("req nil")
	}

	var (
		xl     = xlog.FromContextSafe(ctx)
		uid, _ = reqUID(req)
		// ak, sk, _ = aksk(s.AppdAccessHost, uid)
		ak, sk, err = aksk2(s.qCLI, uid)

		qboxClient = &rpc.Client{
			Client: &http.Client{
				Timeout: time.Second * 5,
				Transport: qboxmac.NewTransport(
					&qboxmac.Mac{AccessKey: ak, SecretKey: []byte(sk)},
					http.DefaultTransport,
				),
			},
		}

		entry struct {
			FH       string            `json:"fh"`
			Hash     string            `json:"hash"`
			Fsize    float32           `json:"fsize"`
			PutTime  int64             `json:"putTime"`
			MimeType string            `json:"mimeType"`
			XqnMeta  map[string]string `json:"x-qn-meta"`
		}
		bucket, key []byte
	)

	if err != nil {
		return httputil.NewError(http.StatusInternalServerError, err.Error())
	}

	{
		_bucket := req.Header.Get("X-Qiniu-Bucket")
		if _bucket == "" {
			return httputil.NewError(http.StatusBadRequest, "need bucket param")
		}
		bucket, err = base64.StdEncoding.DecodeString(_bucket)
		if err != nil {
			return httputil.NewError(http.StatusBadRequest, "decode bucket name error")
		}
	}
	{
		_key := req.Header.Get("X-Qiniu-Key")
		if _key == "" {
			return httputil.NewError(http.StatusBadRequest, "need file key param")
		}
		key, err = base64.StdEncoding.DecodeString(_key)
		if err != nil {
			return httputil.NewError(http.StatusBadRequest, "decode file key error")
		}
	}

	xl.Infof("Bucket:%v,Key:%v", string(bucket), string(key))
	encodedEntryURI := base64.URLEncoding.EncodeToString([]byte(string(bucket) + ":" + string(key)))

	err = qboxClient.CallWith(ctx, &entry, "POST",
		s.RsHost+"/stat/"+encodedEntryURI, "application/x-www-form-urlencoded",
		nil, 0)
	if err != nil {
		return httputil.NewError(http.StatusBadRequest, "get file hash error: "+err.Error())
	}

	encodehash := base64.URLEncoding.EncodeToString([]byte("hash=" + entry.Hash))
	url := s.RsHost + "/chstatus/" + encodedEntryURI + "/status/1/cond/" + encodehash
	err = qboxClient.CallWith(ctx, new(interface{}), "POST", url, "application/x-www-form-urlencoded", nil, 0)
	if err != nil {
		xl.Errorf("failed to disable the resource:%v", err)
		//return nil, httputil.NewError(http.StatusInternalServerError, "failed to disable the resource")
	}

	return err
}

//====

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
