package serving

import (
	"context"
	"io/ioutil"
	"net/http"
	"time"

	"github.com/qiniu/http/httputil.v1"
	"github.com/qiniu/rpc.v3"
	"github.com/qiniu/xlog.v1"
)

type EvalFaceReq struct {
	Data struct {
		URI       string `json:"uri"`
		Attribute struct {
			Pts [][2]int `json:"pts"`
		} `json:"attribute"`
	} `json:"data"`
}

//----------------------------------------------------------------------------//
type FaceFeatureV2 interface {
	Eval(context.Context, EvalFaceReq) ([]byte, error)
}

type _FaceFeatureV2 struct {
	url     string
	timeout time.Duration
	*rpc.Client
}

func NewFaceFeatureV2(conf EvalConfig) _FaceFeatureV2 {
	url := conf.Host + "/v1/eval/facex-feature-v3"
	if conf.URL != "" {
		url = conf.URL
	}
	return _FaceFeatureV2{url: url, timeout: time.Duration(conf.Timeout) * time.Second}
}

func (ff _FaceFeatureV2) Eval(
	ctx context.Context, req EvalFaceReq,
) (bs []byte, err error) {

	var (
		xl     = xlog.FromContextSafe(ctx)
		client *rpc.Client
	)
	if ff.Client == nil {
		client = NewDefaultStubRPCClient(ff.timeout)
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
