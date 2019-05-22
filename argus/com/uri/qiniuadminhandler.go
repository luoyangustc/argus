package uri

import (
	"context"
	"net/http"
	"net/url"
	"regexp"
	"strings"

	"github.com/pkg/errors"
	"qiniu.com/auth/qboxmac.v1"
)

type qiniuAdminHandler struct {
	ioClient *http.Client
	ioHost   string
}

var pathRegex = regexp.MustCompile(`/(.*?)/(.*)`)

func (h *qiniuAdminHandler) Get(ctx context.Context, args Request, opts ...GetOption,
) (resp *Response, err error) {

	uri := args.URI
	var bucket, key, uid string
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
		uid = u.User.Username()
	}

	param := url.Values{}
	param.Add("uid", uid)
	param.Add("bucket", bucket)
	param.Add("key", key)
	req, err := http.NewRequest("POST", h.ioHost+"/adminget/", strings.NewReader(param.Encode()))
	if err != nil {
		return nil, err
	}
	req = req.WithContext(ctx)
	req.Header.Add("Content-Type", "application/x-www-form-urlencoded")
	_resp, err := h.ioClient.Do(req)
	if err != nil {
		return nil, errors.Wrap(err, "ioClient.Do")
	}
	if _resp.StatusCode/100 != 2 {
		defer _resp.Body.Close()
		return nil, responseError(_resp)
	}
	return transResp(_resp), err
}

func (h *qiniuAdminHandler) Names() []string {
	return []string{"qiniu"}
}

// WithAdminAkSk 使用 admin ak sk，可以获取任意用户的qiniu bucket资源
func WithAdminAkSk(adminAk string, adminSk string, ioHost string) Handler {
	return &qiniuAdminHandler{
		ioHost: ioHost,
		ioClient: &http.Client{
			Transport: qboxmac.NewTransport(&qboxmac.Mac{
				AccessKey: adminAk,
				SecretKey: []byte(adminSk),
			}, nil),
		},
	}
}
