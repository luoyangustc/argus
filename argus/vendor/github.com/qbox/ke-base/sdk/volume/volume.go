package volume

import (
	"context"
	"fmt"
	"net/http"

	"github.com/qbox/ke-base/sdk/proto"
	"github.com/qbox/ke-base/sdk/rpc"
	"github.com/qbox/ke-base/sdk/util"
)

type Client interface {
	ListStorageTypes(ctx context.Context) (ret []proto.StorageType, err error)
	PostVolume(ctx context.Context, spaceName string, vol proto.Volume) (ret proto.Volume, err error)
	ListVolume(ctx context.Context, spaceName string) (ret []proto.Volume, err error)
	GetVolume(ctx context.Context, spaceName, volName string) (ret proto.Volume, err error)
	DeleteVolume(ctx context.Context, spaceName, volName string) (err error)
}

type client struct {
	Client rpc.Client
	Host   string
}

func (p *client) ListStorageTypes(ctx context.Context) (ret []proto.StorageType, err error) {
	url := fmt.Sprintf("%s/v1/storagetypes", p.Host)
	err = p.Client.Call(ctx, &ret, "GET", url)
	return
}
func (p *client) PostVolume(ctx context.Context, spaceName string, cm proto.Volume) (ret proto.Volume, err error) {
	url := fmt.Sprintf("%s/v1/projects/%s/volumes", p.Host, spaceName)
	err = p.Client.CallWithJson(ctx, &ret, "POST", url, cm)
	return
}
func (p *client) ListVolume(ctx context.Context, spaceName string) (ret []proto.Volume, err error) {
	url := fmt.Sprintf("%s/v1/projects/%s/volumes", p.Host, spaceName)
	err = p.Client.Call(ctx, &ret, "GET", url)
	return
}
func (p *client) GetVolume(ctx context.Context, spaceName, volName string) (ret proto.Volume, err error) {
	url := fmt.Sprintf("%s/v1/projects/%s/volumes/%s", p.Host, spaceName, volName)
	err = p.Client.Call(ctx, &ret, "GET", url)
	return
}
func (p *client) DeleteVolume(ctx context.Context, spaceName, volName string) (err error) {
	url := fmt.Sprintf("%s/v1/projects/%s/volumes/%s", p.Host, spaceName, volName)
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
