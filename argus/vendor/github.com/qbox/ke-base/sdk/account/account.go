package account

import (
	"fmt"
	"net/http"

	"github.com/qbox/ke-base/sdk/proto"
	"github.com/qbox/ke-base/sdk/rpc"
	"github.com/qbox/ke-base/sdk/util"
	"golang.org/x/net/context"
)

type Client interface {
	GetUserInfo(ctx context.Context, uid string) (userinfo proto.UserInfo, err error)
	GetUserProjects(ctx context.Context, uid string, includeType bool) (projects []*proto.ProjectInfo, err error)
	GetUserRolesOnProject(ctx context.Context, uid, project string) (roles []string, err error)
	AssignRoleToUserOnProject(ctx context.Context, uid, project, role string) error
	ValidateUserRoleOnProject(ctx context.Context, uid, project, role string) error
	DeleteUserRoleOnProject(ctx context.Context, uid, project, role string) error
	GetProjectUsers(ctx context.Context, project string, includeDetailed bool) (users []*proto.UserRoleInfo, err error)
	GetProjectInfo(ctx context.Context, project string) (projectinfo proto.ProjectInfo, err error)
	UpdateProjectInfo(ctx context.Context, project string, info *proto.UpdateProjectInfo) (projectinfo proto.ProjectInfo, err error)
	GetRegions(ctx context.Context) (regions []*proto.RegionInfo, err error)
	GetRegionConfig(ctx context.Context, region string) (config proto.RegionConfig, err error)
	CreateProject(ctx context.Context, opt *proto.CreateProjectOpt) (projectinfo proto.ProjectInfo, err error)
	ValidateUser(ctx context.Context, args *proto.ValidatePassword) error
}

type client struct {
	Client rpc.Client
	Host   string
}

func New(host, token string) Client {
	transport := util.NewAuthTokenTransport(token, http.DefaultTransport)
	return NewWithTransport(host, transport)
}
func NewWithTransport(host string, tr http.RoundTripper) Client {
	c := rpc.Client{&http.Client{Transport: tr}}
	return &client{
		Client: c,
		Host:   util.CleanHost(host) + "/v1",
	}
}

func (cli *client) GetUserInfo(ctx context.Context, uid string) (userinfo proto.UserInfo, err error) {
	url := fmt.Sprintf("%s/users/%s", cli.Host, uid)
	err = cli.Client.Call(ctx, &userinfo, http.MethodGet, url)
	return
}

func (cli *client) GetUserProjects(ctx context.Context, uid string, includeType bool) (projects []*proto.ProjectInfo, err error) {
	url := fmt.Sprintf("%s/users/%s/projects", cli.Host, uid)
	if includeType {
		url += "?includeType=true"
	}
	err = cli.Client.Call(ctx, &projects, http.MethodGet, url)
	return
}

func (cli *client) GetUserRolesOnProject(ctx context.Context, uid, project string) (roles []string, err error) {
	url := fmt.Sprintf("%s/projects/%s/users/%s/roles", cli.Host, project, uid)
	err = cli.Client.Call(ctx, &roles, http.MethodGet, url)
	return
}

func (cli *client) AssignRoleToUserOnProject(ctx context.Context, uid, project, role string) error {
	url := fmt.Sprintf("%s/projects/%s/users/%s/roles/%s", cli.Host, project, uid, role)
	return cli.Client.Call(ctx, nil, http.MethodPut, url)
}

func (cli *client) ValidateUserRoleOnProject(ctx context.Context, uid, project, role string) error {
	url := fmt.Sprintf("%s/projects/%s/users/%s/roles/%s", cli.Host, project, uid, role)
	return cli.Client.Call(ctx, nil, http.MethodHead, url)
}

func (cli *client) DeleteUserRoleOnProject(ctx context.Context, uid, project, role string) error {
	url := fmt.Sprintf("%s/projects/%s/users/%s/roles/%s", cli.Host, project, uid, role)
	return cli.Client.Call(ctx, nil, http.MethodDelete, url)
}

func (cli *client) GetProjectUsers(ctx context.Context, project string, includeDetailed bool) (users []*proto.UserRoleInfo, err error) {
	url := fmt.Sprintf("%s/projects/%s/users", cli.Host, project)
	if includeDetailed {
		url += "?detailed=true"
	}
	err = cli.Client.Call(ctx, &users, http.MethodGet, url)
	return
}

func (cli *client) GetProjectInfo(ctx context.Context, project string) (projectinfo proto.ProjectInfo, err error) {
	url := fmt.Sprintf("%s/projects/%s", cli.Host, project)
	err = cli.Client.Call(ctx, &projectinfo, http.MethodGet, url)
	return
}

func (cli *client) UpdateProjectInfo(ctx context.Context, project string, info *proto.UpdateProjectInfo) (projectinfo proto.ProjectInfo, err error) {
	url := fmt.Sprintf("%s/projects/%s", cli.Host, project)
	err = cli.Client.CallWithJson(ctx, &projectinfo, http.MethodPatch, url, info)
	return
}

func (cli *client) GetRegions(ctx context.Context) (regions []*proto.RegionInfo, err error) {
	url := fmt.Sprintf("%s/regions", cli.Host)
	err = cli.Client.Call(ctx, &regions, http.MethodGet, url)
	return
}

func (cli *client) GetRegionConfig(ctx context.Context, region string) (config proto.RegionConfig, err error) {
	url := fmt.Sprintf("%s/regions/%s/config", cli.Host, region)
	err = cli.Client.Call(ctx, &config, http.MethodGet, url)
	return
}

func (cli *client) CreateProject(ctx context.Context, opt *proto.CreateProjectOpt) (projectinfo proto.ProjectInfo, err error) {
	url := fmt.Sprintf("%s/projects", cli.Host)
	err = cli.Client.CallWithJson(ctx, &projectinfo, http.MethodPost, url, opt)
	return
}

func (cli *client) ValidateUser(ctx context.Context, args *proto.ValidatePassword) error {
	url := fmt.Sprintf("%s/validateuser", cli.Host)
	return cli.Client.CallWithJson(ctx, nil, http.MethodPost, url, args)
}
