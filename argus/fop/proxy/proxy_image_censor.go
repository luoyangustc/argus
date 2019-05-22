package proxy

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/url"
	"strings"
	"time"

	httputil "github.com/qiniu/http/httputil.v1"
	"github.com/qiniu/http/restrpc.v1"
	xlog "github.com/qiniu/xlog.v1"

	ahttp "qiniu.com/argus/argus/com/http"
	"qiniu.com/argus/censor"
	"qiniu.com/argus/censor/biz"
	"qiniu.com/argus/com/proxy/fop"
)

var _ fop.Proxy = ImageCensor{}

type ImageCensor struct {
	URL string
}

func NewImageCensor(_url string) fop.Proxy { return ImageCensor{URL: _url} }

func (p ImageCensor) Post(
	ctx context.Context, req fop.ProxyReq, env *restrpc.Env,
) (interface{}, error) {

	xl := xlog.FromContextSafe(ctx)
	uid, bucket, key, err := reqSourceInfo(env.Req)
	xl.Infof("SrcInfo, %d, %s, %s, %+v", uid, bucket, key, err)

	var types = []string{}
	var isSug = false
	if args := strings.Split(req.Cmd, "/"); len(args) > 1 {
		for i := 1; i < len(args); i++ {
			switch args[i] {
			case "v2":
				isSug = true
			case "pulp", "terror", "politician":
				types = append(types, args[i])
			}
		}
	}

	var uri = req.URL
	if req.URL == "" {
		uri = "data:application/octet-stream;base64," +
			base64.StdEncoding.EncodeToString(req.ReqBody)
		// return nil, httputil.NewError(http.StatusInternalServerError, "unsupport post")
	}

	var req2 = struct {
		Data struct {
			URI string `json:"uri"`
		} `json:"data"`
		Params *struct {
			Type []string `json:"type,omitempty"`
		} `json:"params,omitempty"`
	}{
		Data: struct {
			URI string `json:"uri"`
		}{
			URI: uri,
		},
	}
	if len(types) > 0 {
		req2.Params = &struct {
			Type []string `json:"type,omitempty"`
		}{
			Type: types,
		}
	}

	if isSug { // 新规格流程
		icurl, _ := url.Parse(p.URL)
		icCliFunc := censor.NewImageCensorHTTPClient(
			fmt.Sprintf("%s://%s", icurl.Scheme, icurl.Host), time.Second*60)
		icCli := icCliFunc(uid, 4)

		scenes := make([]biz.Scene, 0, len(types))
		for _, t := range types {
			scenes = append(scenes, biz.Scene(t))
		}

		if len(scenes) <= 0 {
			scenes = []biz.Scene{
				biz.PULP, biz.TERROR, biz.POLITICIAN,
			}
		}

		ret := censor.ImageRecognition(ctx, uri, icCli, scenes, nil)
		if ret.Code >= 300 {
			err := httputil.NewError(ret.Code, ret.Message)
			xl.Errorf("ret, %s, %+v", JsonStr(ret), err)
			return nil, err
		}

		xl.Infof("ret, %s, %+v", JsonStr(ret), nil)
		return ret, nil
	}

	var (
		client = ahttp.NewQiniuStubRPCClient(uint32(req.UID), 4, time.Second*60)
		ret    = new(interface{})
		call   = func(ctx context.Context) error {
			return client.CallWithJson(ctx, ret, "POST", p.URL, req2)
		}
	)

	err = ahttp.CallWithRetry(ctx, []int{530}, []func(context.Context) error{call, call})
	if err != nil {
		xl.Errorf("ret, %s, %+v", JsonStr(ret), err)
		return ret, err
	}

	xl.Infof("ret, %s, %+v", JsonStr(ret), err)
	return ret, err
}

func JsonStr(obj interface{}) string {
	raw, err := json.Marshal(obj)
	if err != nil {
		return ""
	}
	return string(raw)
}
