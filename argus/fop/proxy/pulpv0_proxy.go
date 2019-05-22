package proxy

import (
	"context"
	"encoding/base64"
	"fmt"
	"time"

	"github.com/qiniu/http/restrpc.v1"

	ahttp "qiniu.com/argus/argus/com/http"
	"qiniu.com/argus/com/proxy/fop"
)

type _Pulpv0Ret struct {
	Code      int    `json:"code"`
	Message   string `json:"message"`
	Nonce     string `json:"nonce"`
	TimeStamp int64  `json:"timestamp"`
	Nrop      struct {
		ReviewCount int   `json:"reviewCount"`
		Statistic   []int `json:"statistic"`
		FileList    []struct {
			Result struct {
				Rate   float32 `json:"rate"`
				Label  int     `json:"label"`
				Name   string  `json:"name"`
				Review bool    `json:"review"`
			} `json:"result"`
		} `json:"fileList"`
	} `json:"pulp"`
}

type PulpResp struct {
	Code      int    `json:"code"`
	Message   string `json:"message"`
	Nonce     string `json:"nonce"`
	TimeStamp int64  `json:"timestamp"`
	Pulp      struct {
		Rate   float32 `json:"rate"`
		Label  int     `json:"label"`
		Review bool    `json:"review"`
	} `json:"pulp"`
}

var _ fop.Proxy = Pulpv0Proxy{}

type Pulpv0Proxy struct {
	URL string
}

func NewPulpv0Proxy(_url string) fop.Proxy { return Pulpv0Proxy{URL: _url} }

func (p Pulpv0Proxy) Post(ctx context.Context, req fop.ProxyReq, env *restrpc.Env) (interface{}, error) {

	var (
		uri  = req.URL
		resp PulpResp
	)

	if req.URL == "" {
		uri = "data:application/octet-stream;base64," + base64.StdEncoding.EncodeToString(req.ReqBody)
		// return nil, httputil.NewError(http.StatusInternalServerError, "unsupport post")
	}

	var (
		client = ahttp.NewQiniuStubRPCClient(uint32(req.UID), 0, time.Second*60)
		ret    = new(_Pulpv0Ret)
		call   = func(ctx context.Context) error {
			return client.CallWithJson(ctx, ret, "POST", p.URL,
				struct {
					Image []string `json:"image"`
				}{
					Image: []string{
						uri,
					},
				})
		}
	)

	err := ahttp.CallWithRetry(ctx, []int{530}, []func(context.Context) error{call, call})
	if err != nil {
		return nil, err
	} else if ret.Code != 0 && ret.Code/100 != 2 || len(ret.Nrop.FileList) == 0 {
		err = fmt.Errorf("pulp recognition failed, message:%v,code:%v", ret.Message, ret.Message)
		return nil, err
	}

	resp.Message = ret.Message
	resp.Nonce = ret.Nonce
	resp.TimeStamp = ret.TimeStamp
	resp.Pulp.Label = ret.Nrop.FileList[0].Result.Label
	resp.Pulp.Rate = ret.Nrop.FileList[0].Result.Rate
	resp.Pulp.Review = ret.Nrop.FileList[0].Result.Review

	if resp.Pulp.Review {
		env.W.Header().Set("X-Origin-A", "PULP_Depend,1")
	} else {
		env.W.Header().Set("X-Origin-A", "PULP_Certain,1")
	}
	return resp, nil
}
