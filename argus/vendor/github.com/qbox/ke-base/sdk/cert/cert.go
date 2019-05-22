package cert

import (
	"fmt"
	"net/http"

	"github.com/qbox/ke-base/sdk/proto"
	"github.com/qbox/ke-base/sdk/rpc"
	"github.com/qbox/ke-base/sdk/util"
	"golang.org/x/net/context"
)

type Client interface {
	ListCerts(ctx context.Context, projectName string) (ret []proto.Cert, err error)
	AddCert(ctx context.Context, projectName string, certID string) (err error)
	RemoveCert(ctx context.Context, projectName string, certID string) (err error)
}

type client struct {
	Client rpc.Client
	Host   string
}

func (p *client) ListCerts(ctx context.Context, projectName string) (ret []proto.Cert, err error) {
	url := fmt.Sprintf("%s/v1/projects/%s/certs", p.Host, projectName)
	err = p.Client.Call(ctx, &ret, "GET", url)
	return
}

func (p *client) AddCert(ctx context.Context, projectName string, certID string) (err error) {
	url := fmt.Sprintf("%s/v1/projects/%s/certs/%s", p.Host, projectName, certID)
	err = p.Client.Call(ctx, nil, "POST", url)
	return
}

func (p *client) RemoveCert(ctx context.Context, projectName string, certID string) (err error) {
	url := fmt.Sprintf("%s/v1/projects/%s/certs/%s", p.Host, projectName, certID)
	err = p.Client.Call(ctx, nil, "DELETE", url)
	return
}

func New(host, token string) Client {
	transport := util.NewAuthTokenTransport(token, http.DefaultTransport)
	return NewWithTransport(host, transport)
}

func NewWithTransport(host string, tr http.RoundTripper) Client {
	c := rpc.Client{&http.Client{Transport: tr}}
	return &client{
		Client: c,
		Host:   util.CleanHost(host),
	}
}
