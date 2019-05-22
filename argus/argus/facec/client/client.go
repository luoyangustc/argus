package client

import (
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"context"
	"io/ioutil"

	"github.com/qiniu/http/httputil.v1"
	"github.com/qiniu/xlog.v1"
	"qbox.us/errors"
	"qiniu.com/auth/qiniumac.v1"
	"qiniu.com/argus/argus/monitor"
	"qiniupkg.com/x/rpc.v7"
)

type Config struct {
	Hosts   Hosts
	Timeout time.Duration
	Env     EvalEnv
}

type Client struct {
	hosts  Hosts
	client *rpc.Client
}

type Hosts struct {
	FacexDet     string `json:"facex_det"`
	FacexFeature string `json:"facex_feature"`
	FacexCluster string `json:"facex_cluster"`
}

type EvalEnv struct {
	Uid   uint32
	Utype uint32
}

type EvalTransport struct {
	EvalEnv
	http.RoundTripper
}

func (t EvalTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	req.Header.Set(
		"Authorization",
		fmt.Sprintf("QiniuStub uid=%d&ut=%d", t.Uid, t.Utype),
	)
	return t.RoundTripper.RoundTrip(req)
}

func newQiniuAuthClient(ak, sk string, timeout time.Duration) *rpc.Client {
	return &rpc.Client{
		Client: &http.Client{
			Timeout: timeout,
			Transport: qiniumac.NewTransport(
				&qiniumac.Mac{AccessKey: ak, SecretKey: []byte(sk)},
				http.DefaultTransport,
			),
		},
	}
}

var NewRPCClient = func(env EvalEnv, timeout time.Duration) *rpc.Client {
	client := &rpc.Client{
		Client: &http.Client{
			Timeout: timeout,
			Transport: &EvalTransport{
				EvalEnv:      env,
				RoundTripper: http.DefaultTransport,
			},
		},
	}
	return client
}

////////////////////////////////////////////////////////////////////////////////

func New(cfg Config) *Client {

	return &Client{
		client: &rpc.Client{
			Client: &http.Client{
				Timeout: cfg.Timeout,
			},
		},
		hosts: cfg.Hosts,
	}
}

func dumpJSON(v interface{}) string {
	buf, _ := json.Marshal(v)
	return string(buf)
}

func (c *Client) callWithJson(ctx context.Context, api string, ret interface{}, method, url1 string, param interface{}, env EvalEnv, timeout int64) (err error) {
	xl := xlog.FromContextSafe(ctx)
	start := time.Now()
	defer func(begin time.Time) {
		xl.Xprof("C_"+api, begin, err)
		monitor.InferenceResponseTime(api, err, time.Since(begin))
	}(start)
	cli := NewRPCClient(env, time.Duration(timeout)*time.Second)
	err = cli.CallWithJson(ctx, ret, method, url1, param)
	if err == nil {
		xl.Infof("[CLINET] url:%s useTime:%v param:%s ret:%s err:%#v ", url1, time.Since(start), dumpJSON(param), dumpJSON(ret), err)
	} else {
		xl.Errorf("[CLINET] url:%s useTime:%v param:%s ret:%s err:%#v ", url1, time.Since(start), dumpJSON(param), dumpJSON(ret), err)
	}
	return
}

// old https://github.com/qbox/argus/blob/dev/docs/inference_api/facex.md
// latest https://github.com/qbox/ava/blob/dev/docs/AtServing.api.md#facex_detect
func (c *Client) PostFacexDex(ctx context.Context, args []string, env EvalEnv) (ret []FacexDetResp, err error) {
	var (
		req = make([]struct {
			OP   string `json:"op"`
			Data struct {
				URI string `json:"uri"`
			} `json:"data"`
		}, 0, len(args))
	)
	for _, arg := range args {
		req = append(req,
			struct {
				OP   string `json:"op"`
				Data struct {
					URI string `json:"uri"`
				} `json:"data"`
			}{
				OP: "/v1/eval/facex-detect",
				Data: struct {
					URI string `json:"uri"`
				}{
					URI: arg,
				},
			})
	}

	err = c.callWithJson(ctx, "PostFacexDex", &ret, "POST",
		fmt.Sprintf("%s/v1/batch", c.hosts.FacexDet), // eval/facex-detect", c.hosts.FacexDet)
		req, env, 300)
	return
}

// old https://github.com/qbox/argus/blob/dev/docs/inference_api/facex.md
// latest https://github.com/qbox/ava/blob/dev/docs/AtServing.api.md#facex-feature
func (c *Client) PostFacexFeature1(ctx context.Context, args FacexFeatureReq, env EvalEnv) (ret []byte, err error) {
	url := fmt.Sprintf("%s/v1/eval/facex-feature", c.hosts.FacexFeature)
	xl := xlog.FromContextSafe(ctx)
	start := time.Now()
	defer func(begin time.Time) {
		xl.Xprof("C_PostFacexFeature1", begin, err)
		monitor.InferenceResponseTime("PostFacexFeature1", err, time.Since(begin))
	}(start)

	cli := NewRPCClient(env, time.Second*600)
	rep, err := cli.DoRequestWithJson(ctx, "POST", url, args)
	if err != nil {
		return nil, err
	}
	if rep != nil && rep.StatusCode/100 != 2 {
		xl.Error("PostFacexFeature1 query facex-feature got no 200 response")
		return nil, httputil.NewError(http.StatusInternalServerError, "query facex-feature got no 200 response")
	}
	ret, err = ioutil.ReadAll(rep.Body)
	defer rep.Body.Close()
	if err != nil || len(ret) == 0 {
		xl.Error("PostFacexFeature1 query facex-feature got no 200 response")
		return nil, errors.New("query facex-feature got empty body")
	}
	return
}

// old https://github.com/qbox/argus/blob/dev/docs/inference_api/facex.md
// latest https://github.com/qbox/ava/blob/dev/docs/AtServing.api.md#facex-feature
func (c *Client) PostFacexFeature2(ctx context.Context, args FacexFeatureReq, env EvalEnv) (ret []byte, err error) {
	url := fmt.Sprintf("%s/v1/eval/facex-feature", c.hosts.FacexFeature)
	xl := xlog.FromContextSafe(ctx)
	start := time.Now()
	defer func(begin time.Time) {
		xl.Xprof("C_PostFacexFeature2", begin, err)
		monitor.InferenceResponseTime("PostFacexFeature2", err, time.Since(begin))
	}(start)

	cli := NewRPCClient(env, time.Second*600)
	rep, err := cli.DoRequestWithJson(ctx, "POST", url, args)
	if err != nil {
		return nil, err
	}
	if rep != nil && rep.StatusCode/100 != 2 {
		xl.Error("PostFacexFeature2 query facex-feature got no 200 response")
		return nil, httputil.NewError(http.StatusInternalServerError, "query facex-feature got no 200 response")
	}
	ret, err = ioutil.ReadAll(rep.Body)
	defer rep.Body.Close()
	if err != nil || len(ret) == 0 {
		xl.Error("PostFacexFeature2 query facex-feature got no 200 response")
		return nil, errors.New("query facex-feature got empty body")
	}
	return
}

// old https://github.com/qbox/argus/blob/dev/docs/inference_api/facex.md
// latest https://github.com/qbox/ava/blob/dev/docs/AtServing.api.md#facex-cluster
func (c *Client) PostFacexCluster(ctx context.Context, args FacexClusterReq, env EvalEnv) (ret FacexClusterResp, err error) {
	url := fmt.Sprintf("%s/v1/eval/facex-cluster", c.hosts.FacexCluster)
	err = c.callWithJson(ctx, "PostFacexCluster", &ret, "POST", url, args, env, 1200)
	return
}
