package utility

import (
	"context"
	"io/ioutil"
	"net/http"
	"time"

	"qbox.us/net/httputil"

	"qiniu.com/auth/qiniumac.v1"

	"github.com/qiniu/rpc.v3"
	xlog "github.com/qiniu/xlog.v1"

	ahttp "qiniu.com/argus/argus/com/http"
)

type EvalsConfig struct {
	EvalConfig `json:"default"`
	Evals      map[string]EvalConfig `json:"evals"`
}

func (c EvalsConfig) Get(cmd string) EvalConfig {
	if cc, ok := c.Evals[cmd]; ok {
		if cc.Host == "" {
			cc.Host = c.EvalConfig.Host
		}
		if cc.Timeout == 0 {
			cc.Timeout = c.EvalConfig.Timeout
		}
		return cc
	}
	return c.EvalConfig
}

type EvalConfig struct {
	Host    string        `json:"host"`
	URL     string        `json:"url"`
	Timeout time.Duration `json:"timeout"`
	Auth    struct {
		AK string `json:"ak"`
		SK string `json:"sk"`
	} `json:"auth"`
}

type _EvalEnv struct {
	Uid   uint32
	Utype uint32
}

// type evalTransport struct {
// 	_EvalEnv
// 	http.RoundTripper
// }

// func (t evalTransport) RoundTrip(req *http.Request) (*http.Response, error) {
// 	req.Header.Set(
// 		"Authorization",
// 		fmt.Sprintf("QiniuStub uid=%d&ut=%d", t.Uid, 0), // t.Utype), 特殊设置，避免Argus|Serving同时计量计费
// 	)
// 	return t.RoundTripper.RoundTrip(req)
// }

func newRPCClient(env _EvalEnv, timeout time.Duration) *rpc.Client {
	return ahttp.NewQiniuStubRPCClient(env.Uid, 0, timeout)
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

////////////////////////////////////////////////////////////////////////////////

type _EvalFaceDetectReq struct {
	Data struct {
		URI string `json:"uri"`
	} `json:"data"`
}

type _EvalFaceDetection struct {
	Index int      `json:"index"`
	Class string   `json:"class"`
	Score float32  `json:"score"`
	Pts   [][2]int `json:"pts"`
}

type _EvalFaceDetectResp struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Result  struct {
		Detections []_EvalFaceDetection `json:"detections"`
	} `json:"result"`
}

type iFaceDetect interface {
	Eval(context.Context, _EvalFaceDetectReq, _EvalEnv) (_EvalFaceDetectResp, error)
}

type _FaceDetect struct {
	url     string
	timeout time.Duration
	*rpc.Client
}

func newFaceDetect(host string, timeout time.Duration) _FaceDetect {
	return _FaceDetect{url: host + "/v1/eval/facex-detect", timeout: timeout}
}

func (fd _FaceDetect) Eval(
	ctx context.Context, req _EvalFaceDetectReq, env _EvalEnv,
) (_EvalFaceDetectResp, error) {

	var (
		client *rpc.Client
		resp   _EvalFaceDetectResp
	)
	if fd.Client == nil {
		client = newRPCClient(env, fd.timeout)
	} else {
		client = fd.Client
	}
	err := callRetry(ctx,
		func(ctx context.Context) error {
			return client.CallWithJson(ctx, &resp, "POST", fd.url, &req)
		})
	return resp, err
}

//----------------------------------------------------------------------------//

type _EvalImageReq struct {
	Data struct {
		URI       string `json:"uri"`
		Attribute *struct {
			Pts [][2]int `json:"pts"`
		} `json:"attribute,omitempty"`
	} `json:"data"`
}

type iFeature interface {
	Eval(context.Context, _EvalImageReq, _EvalEnv) ([]byte, error)
}

type _Feature struct {
	url     string
	timeout time.Duration
	*rpc.Client
}

func newFeature(host string, timeout time.Duration) _Feature {
	return _Feature{url: host + "/v1/eval/facex-feature", timeout: timeout}
}

func newFeatureV2(host string, timeout time.Duration) _Feature {
	return _Feature{url: host + "/v1/eval/facex-feature-v2", timeout: timeout}
}

func (f _Feature) Eval(
	ctx context.Context, req _EvalImageReq, env _EvalEnv,
) (bs []byte, err error) {

	var (
		xl     = xlog.FromContextSafe(ctx)
		client *rpc.Client
	)
	if f.Client == nil {
		client = newRPCClient(env, f.timeout)
	} else {
		client = f.Client
	}
	var resp *http.Response
	err = callRetry(ctx,
		func(ctx context.Context) error {
			var err1 error
			resp, err1 = client.DoRequestWithJson(ctx, "POST", f.url, &req)
			return err1
		})
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode/100 != 2 || resp.ContentLength == 0 {
		xl.Errorf(
			"call "+f.url+" error:%v,status code:%v,content length:%v,req:%v",
			err, resp.StatusCode, resp.ContentLength, req,
		)
		return nil, err
	}
	bs, err = ioutil.ReadAll(resp.Body)
	if err != nil {
		xl.Errorf("call "+f.url+",read resp body error:%v", err)
	}
	return bs, err
}

//----------------------------------------------------------------------------//

type iFaceFeatureV2 interface {
	Eval(context.Context, _EvalFaceReq, _EvalEnv) ([]byte, error)
}

type _FaceFeatureV2 struct {
	url     string
	timeout time.Duration
	*rpc.Client
}

func newFaceFeatureV2(host string, timeout time.Duration) _FaceFeatureV2 {
	return _FaceFeatureV2{url: host + "/v1/eval/facex-feature-v2", timeout: timeout}
}

func (ff _FaceFeatureV2) Eval(
	ctx context.Context, req _EvalFaceReq, env _EvalEnv,
) (bs []byte, err error) {

	var (
		xl     = xlog.FromContextSafe(ctx)
		client *rpc.Client
	)
	if ff.Client == nil {
		client = newRPCClient(env, ff.timeout)
	} else {
		client = ff.Client
	}
	var resp *http.Response
	err = callRetry(ctx,
		func(ctx context.Context) error {
			var err1 error
			resp, err1 = client.DoRequestWithJson(ctx, "POST", ff.url, &req)
			return err1
		})
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode/100 != 2 || resp.ContentLength == 0 {
		xl.Errorf(
			"call "+ff.url+" error:%v,status code:%v,content length:%v,req:%v",
			err, resp.StatusCode, resp.ContentLength, req,
		)

		return nil, httputil.NewError(http.StatusInternalServerError, "get feature error")
	}
	bs, err = ioutil.ReadAll(resp.Body)
	if err != nil {
		xl.Errorf("call "+ff.url+",read resp body error:%v", err)
	}
	return bs, err
}

//----------------------------------------------------------------------------//

type _EvalImageGroupSearchReq struct {
	Data []struct {
		URI string `json:"uri"`
	} `json:"data"`
	Params struct {
		Limit int `json:"limit,omitempty"`
	} `json:"params"`
}

type _EvalImageGroupSearchResp struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Result  []struct {
		Index int     `json:"index"`
		Score float32 `json:"score"`
	} `json:"result"`
}

type iImageGroupSearch interface {
	Eval(context.Context, _EvalImageGroupSearchReq, _EvalEnv) (_EvalImageGroupSearchResp, error)
}

type _ImageGroupSearch struct {
	url     string
	timeout time.Duration
	*rpc.Client
}

func newImageGroupSearch(host string, timeout time.Duration) _ImageGroupSearch {
	return _ImageGroupSearch{url: host + "/v1/eval/image-search", timeout: timeout}
}

func (c _ImageGroupSearch) Eval(
	ctx context.Context, req _EvalImageGroupSearchReq, env _EvalEnv,
) (_EvalImageGroupSearchResp, error) {
	var (
		client *rpc.Client

		resp _EvalImageGroupSearchResp
	)
	if c.Client == nil {
		client = newRPCClient(env, c.timeout)
	} else {
		client = c.Client
	}
	err := callRetry(ctx,
		func(ctx context.Context) error {
			return client.CallWithJson(ctx, &resp, "POST", c.url, &req)
		})
	return resp, err
}

//----------------------------------------------------------------------------//

type _EvalFaceSearchReq struct {
	Data struct {
		URI string `json:"uri"`
	} `json:"data"`
}

type _EvalFaceSearchResp struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Result  struct {
		Index  int     `json:"index"`
		Class  string  `jsno:"class"`
		Score  float32 `json:"score"`
		Sample struct {
			URL string   `json:"url"`
			Pts [][2]int `json:"pts"`
			ID  string   `json:"id"`
		} `json:"sample"`
	} `json:"result"`
}
