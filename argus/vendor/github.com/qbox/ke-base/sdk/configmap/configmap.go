package configmap

import (
	"context"
	"fmt"
	"net/http"

	"github.com/qbox/ke-base/sdk/proto"
	"github.com/qbox/ke-base/sdk/rpc"
	"github.com/qbox/ke-base/sdk/util"
)

type Client interface {
	PostConfigMap(ctx context.Context, spaceName string, cm proto.ConfigMap) (ret proto.ConfigMap, err error)
	ListConfigMap(ctx context.Context, spaceName string) (ret []proto.ConfigMap, err error)
	GetConfigMap(ctx context.Context, spaceName, cmName string) (ret proto.ConfigMap, err error)
	PutConfigMap(ctx context.Context, spaceName, cmName string, cm proto.ConfigMap) (ret proto.ConfigMap, err error)
	DeleteConfigMap(ctx context.Context, spaceName, cmName string) (err error)
}

type client struct {
	Client rpc.Client
	Host   string
}

func (p *client) PostConfigMap(ctx context.Context, spaceName string, cm proto.ConfigMap) (ret proto.ConfigMap, err error) {
	url := fmt.Sprintf("%s/v1/projects/%s/configmaps", p.Host, spaceName)
	err = p.Client.CallWithJson(ctx, &ret, "POST", url, cm)
	return
}
func (p *client) ListConfigMap(ctx context.Context, spaceName string) (ret []proto.ConfigMap, err error) {
	url := fmt.Sprintf("%s/v1/projects/%s/configmaps", p.Host, spaceName)
	err = p.Client.Call(ctx, &ret, "GET", url)
	return
}
func (p *client) GetConfigMap(ctx context.Context, spaceName, cmName string) (ret proto.ConfigMap, err error) {
	url := fmt.Sprintf("%s/v1/projects/%s/configmaps/%s", p.Host, spaceName, cmName)
	err = p.Client.Call(ctx, &ret, "GET", url)
	return
}
func (p *client) PutConfigMap(ctx context.Context, spaceName, cmName string, cm proto.ConfigMap) (ret proto.ConfigMap, err error) {
	url := fmt.Sprintf("%s/v1/projects/%s/configmaps/%s", p.Host, spaceName, cmName)
	err = p.Client.CallWithJson(ctx, nil, "PUT", url, cm)
	return
}
func (p *client) DeleteConfigMap(ctx context.Context, spaceName, cmName string) (err error) {
	url := fmt.Sprintf("%s/v1/projects/%s/configmaps/%s", p.Host, spaceName, cmName)
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
