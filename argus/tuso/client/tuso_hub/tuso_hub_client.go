package tuso_hub

import (
	"context"
	"fmt"
	"net/http"
	"strconv"
	"time"

	"github.com/qiniu/rpc.v1"
	"github.com/qiniu/xlog.v1"
	"qiniu.com/argus/tuso/proto"
	authstub "qiniu.com/auth/authstub.v1"
)

type Config struct {
	Host          string `json:"host"`
	TimeoutSecond int    `json:"timeout_second"`
}

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

func NewInternalHack(uid uint32, host string, timeout time.Duration) *InternalClient {
	client := &rpc.Client{
		Client: &http.Client{
			Transport: &mockUser{uid: uid},
			Timeout:   timeout,
		},
	}
	return &InternalClient{client: client, host: host}
}

type Client struct {
	client  *rpc.Client
	host    string
	version string
}

var _ proto.UserApi = new(Client)

type InternalClient struct {
	client *rpc.Client
	host   string
}

var _ proto.InternalApi = new(InternalClient)

func (c *Client) PostImage(ctx context.Context, req *proto.PostImageReq, env *authstub.Env) (resp *proto.PostImageResp, err error) {
	xl := xlog.FromContextSafe(ctx)
	url := fmt.Sprintf("%s/%s/image", c.host, c.version)
	err = c.client.CallWithJson(xl, &resp, url, req)
	return
}

func (c *Client) PostSearchJob(ctx context.Context, req *proto.PostSearchJobReq, env *authstub.Env) (resp *proto.PostSearchJobResp, err error) {
	xl := xlog.FromContextSafe(ctx)
	url := fmt.Sprintf("%s/%s/search/job", c.host, c.version)
	err = c.client.CallWithJson(xl, &resp, url, req)
	return
}

func (c *Client) GetSearchJob(ctx context.Context, req *proto.GetSearchJobReq, env *authstub.Env) (resp *proto.GetSearchJobResp, err error) {
	xl := xlog.FromContextSafe(ctx)
	url := fmt.Sprintf("%s/%s/search/job?job_id=%s", c.host, c.version, req.JobID)
	err = c.client.GetCall(xl, &resp, url)
	return
}

func (c *Client) PostHub(ctx context.Context, req *proto.PostHubReq, env *authstub.Env) (err error) {
	xl := xlog.FromContextSafe(ctx)
	url := fmt.Sprintf("%s/%s/hub", c.host, c.version)
	err = c.client.CallWithJson(xl, nil, url, req)
	return
}

func (c *Client) GetHubs(ctx context.Context, req *proto.GetHubsReq, env *authstub.Env) (resp *proto.GetHubsResp, err error) {
	xl := xlog.FromContextSafe(ctx)
	url := fmt.Sprintf("%s/%s/hubs", c.host, c.version)
	err = c.client.GetCall(xl, &resp, url)
	return
}

func (c *Client) GetHub(ctx context.Context, req *proto.GetHubReq, env *authstub.Env) (resp *proto.GetHubResp, err error) {
	xl := xlog.FromContextSafe(ctx)
	url := fmt.Sprintf("%s/%s/hub?hub=%s", c.host, c.version, req.Hub)
	err = c.client.GetCall(xl, &resp, url)
	return
}

func (c *InternalClient) GetHubInfo(ctx context.Context, req *proto.GetHubInfoReq, env *authstub.Env) (resp *proto.GetHubInfoResp, err error) {
	xl := xlog.FromContextSafe(ctx)
	url := fmt.Sprintf("%s/hub_admin/hub/info?name=%s&version=%s", c.host, req.HubName, strconv.Itoa(req.Version))
	err = c.client.GetCall(xl, &resp, url)
	return
}

func (c *InternalClient) GetFilemetaInfo(ctx context.Context, req *proto.GetFileMetaInfoReq, env *authstub.Env) (resp *proto.GetFileMetaInfoResp, err error) {
	xl := xlog.FromContextSafe(ctx)
	url := fmt.Sprintf("%s/hub_admin/filemeta/info?name=%s&index=%s&offset=%s", c.host, req.HubName, strconv.Itoa(req.FeatureFileIndex), strconv.Itoa(req.FeatureFileOffset))
	err = c.client.GetCall(xl, &resp, url)
	return
}
