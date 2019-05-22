package domain

import (
	"context"
	"fmt"
	"net/http"

	"github.com/qbox/ke-base/sdk/proto"
	"github.com/qbox/ke-base/sdk/rpc"
	"github.com/qbox/ke-base/sdk/util"
)

type Client interface {
	PostDomain(ctx context.Context, domain proto.Domain) (ret proto.Domain, err error)
	DelDomain(ctx context.Context, opt proto.DelOption) (err error)
	ListDomain(ctx context.Context, opt proto.ListOption) (ret []proto.Domain, err error)
	PatchDomain(ctx context.Context, opt proto.PatchDomainOption) (ret proto.Domain, err error)
}

type client struct {
	Client rpc.Client
	Host   string
}

func (p *client) PostDomain(ctx context.Context, domain proto.Domain) (ret proto.Domain, err error) {
	url := fmt.Sprintf("%s/v1/projects/%s/domains", p.Host, domain.ProjectName)
	err = p.Client.CallWithJson(ctx, &ret, "POST", url, domain)
	return
}

func (p *client) PatchDomain(ctx context.Context, opt proto.PatchDomainOption) (ret proto.Domain, err error) {
	url := fmt.Sprintf("%s/v1/projects/%s/domains", p.Host, opt.ProjectName)
	err = p.Client.CallWithJson(ctx, &ret, "PATCH", url, opt)
	return
}

func (p *client) DelDomain(ctx context.Context, opt proto.DelOption) (err error) {
	url := fmt.Sprintf("%s/v1/projects/%s/domains?%s", p.Host, opt.ProjectName, opt.ToQuery())
	err = p.Client.Call(ctx, nil, "DELETE", url)
	return
}

func (p *client) ListDomain(ctx context.Context, opt proto.ListOption) (ret []proto.Domain, err error) {
	url := fmt.Sprintf("%s/v1/projects/%s/domains?%s", p.Host, opt.ProjectName, opt.ToQuery())
	err = p.Client.Call(ctx, &ret, "GET", url)
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
