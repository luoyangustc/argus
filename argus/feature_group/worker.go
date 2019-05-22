package feature_group

import (
	"context"
	"encoding/base64"
	"fmt"
	"io/ioutil"
	"net/http"
	"time"

	"github.com/qiniu/http/httputil.v1"
	"github.com/qiniu/xlog.v1"
	"qiniu.com/auth/authstub.v1"

	ahttp "qiniu.com/argus/argus/com/http"
)

type Worker struct {
	*Memory
}

type SearchKey struct {
	Hid     HubID      `json:"hid"`
	Version HubVersion `json:"version"`
	From    int        `json:"from"`
	To      int        `json:"to"`
}

func (key SearchKey) Key() string {
	return fmt.Sprintf("%s:%d_%d-%d", key.Hid, key.Version, key.From, key.To)
}

type SearchReq struct {
	Key SearchKey `json:"key"`

	Threshold float32 `json:"threshold"`
	Limit     int     `json:"limit"`

	Features string `json:"features"` // always Little-Endian in worker.
	Length   int    `json:"length"`
}

// POST /search
func (w Worker) PostSearch(
	ctx context.Context,
	req *SearchReq,
	env *authstub.Env,
) ([]SearchResult, error) {

	xl, ok := xlog.FromContext(ctx)
	if !ok {
		xl = xlog.NewWithReq(env.Req)
		ctx = xlog.NewContext(ctx, xl)
	}

	var err error
	_RequestGauge("search").Inc()
	defer func(t1 time.Time) {
		var code = 200
		if err != nil {
			code = httputil.DetectCode(err)
		}
		_ResponseTimeHistogram("search", code).Observe(float64(time.Since(t1) / time.Second))
	}(time.Now())

	bs, err := base64.StdEncoding.DecodeString(req.Features)
	if err != nil {
		return nil, err
	}

	ret, err := w.Memory.Search(ctx, req.Key, bs, uint64(req.Length*4), req.Threshold, req.Limit)
	xl.Infof("%#v %v", ret, err)
	if err != nil {
		return nil, err
	}

	for i, _ := range ret {
		for j, _ := range ret[i].Items {
			ret[i].Items[j].Version = req.Key.Version
			ret[i].Items[j].Index += req.Key.From
		}
	}
	return ret, nil
}

//----------------------------------------------------------------------------//

type HubFeatureFetch struct {
	URL string
}

func (hub HubFeatureFetch) Fetch(ctx context.Context, _key Key) ([]byte, error) {
	var (
		xl     = xlog.FromContextSafe(ctx)
		client = ahttp.NewQiniuStubRPCClient(1, 0, time.Second*60)

		k    = _key.(SearchKey)
		resp *http.Response
		f    = func(ctx context.Context) error {
			var err1 error
			resp, err1 = client.DoRequestWithJson(ctx, "POST", hub.URL, k)
			return err1
		}
	)

	err := ahttp.CallWithRetry(ctx, []int{530}, []func(context.Context) error{f, f})
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode/100 != 2 || resp.ContentLength == 0 {
		xl.Errorf(
			"call "+hub.URL+" error:%v,status code:%v,content length:%v,req:%v",
			err, resp.StatusCode, resp.ContentLength, k,
		)
		return nil, err
	}
	bs, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		xl.Errorf("call "+hub.URL+",read resp body error:%v", err)
		return nil, err
	}
	return bs, nil
}
