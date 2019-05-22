package job_gate

import (
	"context"
	"fmt"
	"net/http"
	"time"

	"github.com/qiniu/rpc.v1"
	"github.com/qiniu/xlog.v1"
	"qiniu.com/argus/tuso/proto"
	authstub "qiniu.com/auth/authstub.v1"
)

func New(tr http.RoundTripper, host string, timeout time.Duration) *Client {
	client := &rpc.Client{
		Client: &http.Client{
			Transport: tr,
			Timeout:   timeout,
		},
	}
	return &Client{client: client, host: host, version: "v1"}
}

type mockUser struct {
	uid uint32
}

func (m *mockUser) RoundTrip(req *http.Request) (resp *http.Response, err error) {
	req.Header.Set("Authorization", fmt.Sprintf("QiniuStub uid=%v&ut=0", m.uid))
	return http.DefaultTransport.RoundTrip(req)
}

func NewHack(uid uint32, host string, timeout time.Duration) *Client {
	client := &rpc.Client{
		Client: &http.Client{
			Transport: &mockUser{uid: uid},
			Timeout:   timeout,
		},
	}
	return &Client{client: client, host: host, version: "v1"}
}

type Client struct {
	client  *rpc.Client
	host    string
	version string
}

type PostSubmitTusoSearchReq struct {
	Request proto.PostSearchJobReqJob `json:"request"`
}

type PostSubmitTusoSearchResp struct {
	JobID string `json:"job_id"`
}

func (c *Client) PostSubmitTusoSearch(ctx context.Context, req *PostSubmitTusoSearchReq, env *authstub.Env) (resp *proto.PostSearchJobResp, err error) {
	xl := xlog.FromContextSafe(ctx)
	url := fmt.Sprintf("%s/%s/submit/tuso-search", c.host, c.version)
	err = c.client.CallWithJson(xl, &resp, url, req)
	return
}

type GetQueryTusoSearchResp struct {
	Response proto.GetSearchJobResp `json:"response"`
}

func (c *Client) GetQueryTusoSearch(ctx context.Context, req *proto.GetSearchJobReq, env *authstub.Env) (resp *GetQueryTusoSearchResp, err error) {
	xl := xlog.FromContextSafe(ctx)
	url := fmt.Sprintf("%s/%s/query/tuso-search/%s", c.host, c.version, req.JobID)
	err = c.client.GetCall(xl, &resp, url)
	return
}
