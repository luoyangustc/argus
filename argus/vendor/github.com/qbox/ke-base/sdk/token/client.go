package token

import (
	"fmt"
	"net/http"

	"github.com/qbox/ke-base/sdk/rpc"
	"golang.org/x/net/context"
)

func NewUserTokenClient(host, username, password string) UserClient {
	return &userClient{
		Host:     host,
		Username: username,
		Password: password,
		Client:   rpc.Client{Client: http.DefaultClient},
	}
}

type UserClient interface {
	CreateToken(ctx context.Context) (ret UserToken, err error)
	WithOnlyLogin() UserClient
}

type userClient struct {
	Host     string
	Username string
	Password string
	Client   rpc.Client

	onlyLogin bool
}

type createUserTokenOpt struct {
	Username  string `json:"name"`
	Password  string `json:"password"`
	OnlyLogin bool   `json:"onlylogin,omitempty"`
}

func (p *userClient) CreateToken(ctx context.Context) (ret UserToken, err error) {
	url := fmt.Sprintf("%s/v1/usertoken", p.Host)
	err = p.Client.CallWithJson(ctx, &ret, "POST", url, &createUserTokenOpt{
		Username:  p.Username,
		Password:  p.Password,
		OnlyLogin: p.onlyLogin,
	})
	return
}

func (p *userClient) WithOnlyLogin() UserClient {
	p.onlyLogin = true
	return p
}

func NewProjectTokenClient(host string, trans http.RoundTripper) (ret ProjectClient) {
	if trans == nil {
		trans = http.DefaultTransport
	}

	return &projectClient{
		Host:   host,
		Client: rpc.Client{&http.Client{Transport: trans}},
	}
}

type ProjectClient interface {
	CreateToken(ctx context.Context, projectName string) (ret ProjectToken, err error)
}

type projectClient struct {
	Host   string
	Client rpc.Client
}

func (p *projectClient) CreateToken(ctx context.Context, projectName string) (ret ProjectToken, err error) {
	url := fmt.Sprintf("%s/v1/projects/%s/token", p.Host, projectName)
	err = p.Client.Call(ctx, &ret, "GET", url)
	return
}
