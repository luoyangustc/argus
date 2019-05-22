package uri

import (
	"context"
	"net/http"
	"net/url"

	"github.com/pkg/errors"
	"github.com/qiniu/api/conf"
	"qbox.us/api/v2/rs"
	"qiniu.com/auth/qboxmac.v1"
)

type qiniuHandler struct {
	rsHost   string
	client   *http.Client
	rsClient rs.Service
}

func (h *qiniuHandler) Get(ctx context.Context, args Request, opts ...GetOption,
) (resp *Response, err error) {

	uri := args.URI
	var bucket, key string
	{
		// u: Scheme: "qiniu", Host: "z0", Path: "/test/1.png",
		u, err := url.Parse(uri)
		if err != nil {
			return nil, errors.Wrap(err, "url.Parse")
		}
		subStr := pathRegex.FindStringSubmatch(u.Path)
		if len(subStr) != 3 {
			return nil, ErrBadUri
		}
		bucket = subStr[1]
		key = subStr[2]
	}
	encodeKey := bucket + ":" + key
	getRet, _, err := h.rsClient.Get(encodeKey, "")
	if err != nil {
		return nil, errors.Wrap(err, "rs1.Get")
	}
	return getHTTP(ctx, h.client, getRet.URL, "")
}

func (h *qiniuHandler) Names() []string {
	return []string{"qiniu"}
}

func getHTTP(ctx context.Context, client *http.Client, url string, host string) (resp *Response, err error) {
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return nil, errors.Wrap(err, "getHTTP http.NewRequest")
	}
	if host != "" {
		req.URL.Host = host
	}
	req = req.WithContext(ctx)
	_resp, err := client.Do(req)
	if err != nil {
		return nil, errors.Wrap(err, "getHTTP client.Do")
	}
	if _resp.StatusCode/100 != 2 {
		defer _resp.Body.Close()
		return nil, responseError(_resp)
	}
	return transResp(_resp), err
}

// WithUserAkSk 使用用户级别 ak sk, 可以获取这个用户的qiniu bucket资源，uri里面的uid将被忽略
func WithUserAkSk(ak string, sk string, rsHost string) Handler {
	conf.RS_HOST = rsHost
	return &qiniuHandler{
		rsHost: rsHost,
		client: http.DefaultClient,
		rsClient: rs.New(qboxmac.NewTransport(&qboxmac.Mac{
			AccessKey: ak,
			SecretKey: []byte(sk),
		}, nil)),
	}
}
