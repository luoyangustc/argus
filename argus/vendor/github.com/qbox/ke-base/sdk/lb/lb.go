package lb

import (
	"context"
	"fmt"
	"net/http"

	"github.com/qbox/ke-base/sdk/proto"
	"github.com/qbox/ke-base/sdk/rpc"
	"github.com/qbox/ke-base/sdk/util"
)

type Client interface {
	PostAlb(ctx context.Context, alb proto.Alb) (ret proto.Alb, err error)
	PatchAlb(ctx context.Context, opt proto.PatchAlbOption) (ret proto.Alb, err error)
	DelAlb(ctx context.Context, opt proto.DelOption) (err error)
	ListAlb(ctx context.Context, opt proto.ListOption) (ret []proto.Alb, err error)
	ListAlbAdmin(ctx context.Context, opt proto.ListOption) (ret []proto.Alb, err error)

	PostBackendRule(ctx context.Context, rule proto.BackendRule) (proto.BackendRule, error)
	PatchBackendRule(ctx context.Context, opt proto.PatchRuleOption) (proto.BackendRule, error)
	DelBackendRule(ctx context.Context, opt proto.DelOption) error
	ListBackendRule(ctx context.Context, opt proto.ListOption) ([]proto.BackendRule, error)

	PostTlb(ctx context.Context, projectName string, args proto.CreateTlbArgs) (ret proto.Tlb, err error)
	PutTlb(ctx context.Context, projectName string, tlbname string, args proto.UpdateTlbArgs) (ret proto.Tlb, err error)
	DelTlb(ctx context.Context, projectName string, tlbname string) (err error)
	GetTlb(ctx context.Context, projectName string, tlbname string) (ret proto.Tlb, err error)
	ListTlb(ctx context.Context, projectName string, args proto.ListTlbArgs) (ret []proto.Tlb, err error)
}

type client struct {
	Client rpc.Client
	Host   string
}

func (p *client) PostAlb(ctx context.Context, alb proto.Alb) (ret proto.Alb, err error) {
	url := fmt.Sprintf("%s/v1/projects/%s/albs", p.Host, alb.ProjectName)
	err = p.Client.CallWithJson(ctx, &ret, "POST", url, alb)
	return
}

func (p *client) PatchAlb(ctx context.Context,
	opt proto.PatchAlbOption) (ret proto.Alb, err error) {
	url := fmt.Sprintf("%s/v1/projects/%s/albs", p.Host, opt.ProjectName)
	err = p.Client.CallWithJson(ctx, &ret, "PATCH", url, opt)
	return
}

func (p *client) DelAlb(ctx context.Context, opt proto.DelOption) (err error) {
	url := fmt.Sprintf("%s/v1/projects/%s/albs?%s", p.Host, opt.ProjectName, opt.ToQuery())
	err = p.Client.Call(ctx, nil, "DELETE", url)
	return
}

func (p *client) ListAlb(ctx context.Context, opt proto.ListOption) (ret []proto.Alb, err error) {
	url := fmt.Sprintf("%s/v1/projects/%s/albs?%s", p.Host, opt.ProjectName, opt.ToQuery())
	err = p.Client.Call(ctx, &ret, "GET", url)
	return
}

func (p *client) ListAlbAdmin(ctx context.Context, opt proto.ListOption) (ret []proto.Alb, err error) {
	url := fmt.Sprintf("%s/v1/admin/albs?%s", p.Host, opt.ToQuery())
	err = p.Client.Call(ctx, &ret, "GET", url)
	return
}

func (p *client) PostBackendRule(ctx context.Context,
	rule proto.BackendRule) (ret proto.BackendRule, err error) {
	url := fmt.Sprintf("%s/v1/projects/%s/albrules", p.Host, rule.ProjectName)
	err = p.Client.CallWithJson(ctx, &ret, "POST", url, rule)
	return
}

func (p *client) PatchBackendRule(ctx context.Context,
	opt proto.PatchRuleOption) (ret proto.BackendRule, err error) {
	url := fmt.Sprintf("%s/v1/projects/%s/albrules", p.Host, opt.ProjectName)
	err = p.Client.CallWithJson(ctx, &ret, "PATCH", url, opt)
	return
}

func (p *client) DelBackendRule(ctx context.Context, opt proto.DelOption) (err error) {
	url := fmt.Sprintf("%s/v1/projects/%s/albrules?%s", p.Host, opt.ProjectName, opt.ToQuery())
	err = p.Client.Call(ctx, nil, "DELETE", url)
	return
}

func (p *client) ListBackendRule(ctx context.Context,
	opt proto.ListOption) (ret []proto.BackendRule, err error) {
	url := fmt.Sprintf("%s/v1/projects/%s/albrules?%s", p.Host, opt.ProjectName, opt.ToQuery())
	err = p.Client.Call(ctx, &ret, "GET", url)
	return
}

func (p *client) PostTlb(ctx context.Context, projectName string, args proto.CreateTlbArgs) (ret proto.Tlb, err error) {
	url := fmt.Sprintf("%s/v1/projects/%s/tlbs", p.Host, projectName)
	err = p.Client.CallWithJson(ctx, &ret, "POST", url, args)
	return
}

func (p *client) PutTlb(ctx context.Context, projectName string, tlbname string, args proto.UpdateTlbArgs) (ret proto.Tlb, err error) {
	url := fmt.Sprintf("%s/v1/projects/%s/tlbs/%s", p.Host, projectName, tlbname)
	err = p.Client.CallWithJson(ctx, &ret, "PUT", url, args)
	return
}

func (p *client) DelTlb(ctx context.Context, projectName string, tlbname string) (err error) {
	url := fmt.Sprintf("%s/v1/projects/%s/tlbs/%s", p.Host, projectName, tlbname)
	err = p.Client.Call(ctx, nil, "DELETE", url)
	return
}

func (p *client) GetTlb(ctx context.Context, projectName string, tlbname string) (ret proto.Tlb, err error) {
	url := fmt.Sprintf("%s/v1/projects/%s/tlbs/%s", p.Host, projectName, tlbname)
	err = p.Client.Call(ctx, &ret, "GET", url)
	return
}

func (p *client) ListTlb(ctx context.Context, projectName string, args proto.ListTlbArgs) (ret []proto.Tlb, err error) {
	url := fmt.Sprintf("%s/v1/projects/%s/tlbs?service=%s", p.Host, projectName, args.ServiceName)
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
