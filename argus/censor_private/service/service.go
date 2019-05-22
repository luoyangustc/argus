package service

import (
	"context"

	"qiniu.com/argus/censor_private/auth"
	"qiniu.com/argus/censor_private/dao"
	"qiniu.com/argus/censor_private/job"
	"qiniu.com/argus/censor_private/proto"
	"qiniu.com/argus/censor_private/util"

	"github.com/pkg/errors"
	restrpc "github.com/qiniu/http/restrpc.v1"
	xlog "github.com/qiniu/xlog.v1"
)

type IService interface {
	// 审核接口
	PostCensorEntries(context.Context, *CensorEntriesReq, *restrpc.Env) (*CensorEntriesResp, error)
	PostCensorUpdateEntries(context.Context, *CensorEntryReq, *restrpc.Env) error
	PostCensorEntriesDownload(context.Context, *dao.EntryFilter, *restrpc.Env)

	// 监控接口
	PostSetAdd(context.Context, *SetAddReq, *restrpc.Env) (*SetAddResp, error)
	PostSetUpload(context.Context, *restrpc.Env) (*SetAddResp, error)
	PostSet_Start(context.Context, *struct{ CmdArgs []string }, *restrpc.Env) error
	PostSet_Stop(context.Context, *struct{ CmdArgs []string }, *restrpc.Env) error
	PostSet_Update(context.Context, *SetUpdateReq, *restrpc.Env) error
	GetSets(context.Context, *SetsReq, *restrpc.Env) (*SetsResp, error)
	GetSet_History(context.Context, *struct{ CmdArgs []string }, *restrpc.Env) (*SetsHistoryResp, error)

	// 用户管理接口
	PostUserAdd(context.Context, *UserAddReq, *restrpc.Env) error
	PostUserDelete(context.Context, *UserDelReq, *restrpc.Env) error
	PostUserUpdate(context.Context, *UserUpdateReq, *restrpc.Env) error
	PostUserPassword(context.Context, *UserPwdReq, *restrpc.Env) error
	GetUsers(context.Context, *UsersReq, *restrpc.Env) (*UsersResp, error)

	// 登录相关接口
	PostLogin(context.Context, *LoginReq, *restrpc.Env) (*LoginResp, error)
	PostLogout(context.Context, *restrpc.Env) error
	GetConfig(context.Context, *restrpc.Env) (*ConfigResp, error)

	//资源相关接口
	PostResources_(context.Context, *struct {
		CmdArgs []string
		Urls    []string `json:"urls"`
	}, *restrpc.Env) error
}

type ServiceConfig struct {
	Scenes    []proto.Scene
	MimeTypes []proto.MimeType
}

type Service struct {
	config     *ServiceConfig
	session    *auth.Auth
	dispatcher *job.Dispatcher
	fileSaver  util.FileSaver
	proxy      URIProxy
}

func NewService(conf *ServiceConfig, session *auth.Auth, dispatcher *job.Dispatcher,
	fileSaver util.FileSaver, proxy URIProxy) (IService, error) {
	if len(conf.Scenes) == 0 {
		// if not set, use all scenes
		conf.Scenes = proto.ValidScenes
	} else {
		for _, v := range conf.Scenes {
			if !v.IsValid() {
				return nil, errors.Errorf("invalid scene:", v)
			}
		}
	}

	if len(conf.MimeTypes) == 0 {
		// if not set, use all mimetypes
		conf.MimeTypes = proto.ValidMimeTypes
	} else {
		for _, v := range conf.MimeTypes {
			if !v.IsValid() {
				return nil, errors.Errorf("invalid mimetype:", v)
			}
		}
	}
	return &Service{
		config:     conf,
		session:    session,
		dispatcher: dispatcher,
		fileSaver:  fileSaver,
		proxy:      proxy,
	}, nil
}

type LoginReq struct {
	Id       string `json:"id"`
	Password string `json:"password"`
}
type LoginResp struct {
	Id    string       `json:"id"`
	Roles []proto.Role `json:"roles"`
}

func (s *Service) PostLogin(
	ctx context.Context,
	req *LoginReq,
	env *restrpc.Env,
) (*LoginResp, error) {
	xl := xlog.FromContextSafe(ctx)
	if len(req.Id) == 0 {
		xl.Error("empty username")
		return nil, proto.ErrEmptyUsername
	}
	if len(req.Password) == 0 {
		xl.Error("empty password")
		return nil, proto.ErrEmptyPassword
	}
	user, err := dao.UserDao.Find(req.Id)
	if err != nil {
		xl.Errorf("dao.UserDao.Find(%s): %v", req.Id, err)
		return nil, err
	}

	if user.Password != util.Sha1(req.Password) {
		xl.Errorf("incorrect password")
		return nil, proto.ErrIncorrectPwd
	}

	s.session.Login(env.Req, env.W, user)

	resp := LoginResp{Id: req.Id, Roles: user.Roles}

	dao.RoleCache.Set(user.Id, user.Roles)
	return &resp, nil
}

func (s *Service) PostLogout(
	ctx context.Context,
	env *restrpc.Env,
) error {
	s.session.Logout(env.Req, env.W)
	return nil
}

type ConfigResp struct {
	Scenes    []proto.Scene    `json:"scenes"`
	MimeTypes []proto.MimeType `json:"mime_types"`
}

func (s *Service) GetConfig(
	ctx context.Context,
	env *restrpc.Env,
) (*ConfigResp, error) {
	resp := &ConfigResp{
		Scenes:    s.config.Scenes,
		MimeTypes: s.config.MimeTypes,
	}
	return resp, nil
}

func (s *Service) PostResources_(
	ctx context.Context,
	req *struct {
		CmdArgs []string
		Urls    []string `json:"urls"`
	},
	env *restrpc.Env) error {

	xl := xlog.FromContextSafe(ctx)
	resource := req.CmdArgs[0]
	if len(resource) == 0 {
		xl.Error("empty resource")
		return proto.ErrEmptyResource
	}
	if len(req.Urls) == 0 {
		xl.Errorf("empty urls")
		return proto.ErrEmptyUrls
	}

	filter := &dao.SetFilter{
		Uri:    resource,
		Type:   proto.SetTypeMonitorPassive,
		Status: proto.SetStatusRunning,
	}
	set, err := dao.SetDao.Query(filter)
	if err != nil {
		xl.Errorf("setDao entryDao.Query(%#v): %v", filter, err)
		return err
	}
	if len(set) == 0 || len(set) > 1 {
		xl.Errorf("invalid resource")
		return proto.ErrInvalidResource
	}

	// insert in db
	count := dao.InsertEntries(ctx, set[0].Id, set[0].MimeTypes, req.Urls)
	if count > 0 {
		s.dispatcher.Notify(set[0].Id)
		xl.Infof("set(%s) collect %d entries", set[0].Id, count)
	}

	return nil
}
