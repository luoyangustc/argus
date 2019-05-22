package service

import (
	"context"

	restrpc "github.com/qiniu/http/restrpc.v1"
	xlog "github.com/qiniu/xlog.v1"
	"gopkg.in/mgo.v2/bson"
	"qiniu.com/argus/censor_private/dao"
	"qiniu.com/argus/censor_private/proto"
	"qiniu.com/argus/censor_private/util"
)

type UserAddReq struct {
	Id       string       `json:"id"`
	Password string       `json:"password"`
	Desc     string       `json:"desc"`
	Roles    []proto.Role `json:"roles"`
}

func (_ *Service) PostUserAdd(
	ctx context.Context,
	req *UserAddReq,
	env *restrpc.Env,
) error {
	xl := xlog.FromContextSafe(ctx)

	if len(req.Id) == 0 {
		xl.Error("empty username")
		return proto.ErrEmptyUsername
	}

	if len(req.Password) == 0 {
		xl.Error("empty password")
		return proto.ErrEmptyPassword
	}

	if len(req.Roles) == 0 {
		xl.Error("empty roles")
		return proto.ErrEmptyRoles
	}

	for _, r := range req.Roles {
		if r == proto.RoleAdmin {
			xl.Error("cannot create admin")
			return proto.ErrCannotCreateAdmin
		}
		if !r.IsValid() {
			xl.Error("invalid role")
			return proto.ErrInvalidRole
		}
	}

	user, _ := dao.UserDao.Find(req.Id)
	if user != nil {
		xl.Errorf("user exist : %s", req.Id)
		return proto.ErrUserExist
	}

	user = &proto.User{
		Id:       req.Id,
		Desc:     req.Desc,
		Password: util.Sha1(req.Password),
		Roles:    req.Roles,
	}
	err := dao.UserDao.Insert(user)
	if err != nil {
		xl.Errorf("userDao.Insert(%#v): %v", user, err)
		return err
	}

	return nil
}

type UserDelReq struct {
	Id string `json:"id"`
}

func (_ *Service) PostUserDelete(
	ctx context.Context,
	req *UserDelReq,
	env *restrpc.Env,
) error {
	xl := xlog.FromContextSafe(ctx)

	if len(req.Id) == 0 {
		xl.Error("empty username")
		return proto.ErrEmptyUsername
	}

	user, err := dao.UserDao.Find(req.Id)
	if err != nil {
		xl.Errorf("userDao.Find(%s): %v", req.Id, err)
		return err
	}

	for _, r := range user.Roles {
		if r == proto.RoleAdmin {
			xl.Error("cannot delete admin")
			return proto.ErrCannotDeleteAdmin
		}
	}

	err = dao.UserDao.Remove(req.Id)
	if err != nil {
		xl.Errorf("userDao.Remove(%#v): %v", user, err)
		return err
	}
	dao.RoleCache.Remove(user.Id)
	return nil
}

type UserUpdateReq struct {
	Id    string       `json:"id"`
	Desc  string       `json:"desc"`
	Roles []proto.Role `json:"roles"`
}

func (_ *Service) PostUserUpdate(
	ctx context.Context,
	req *UserUpdateReq,
	env *restrpc.Env,
) error {
	xl := xlog.FromContextSafe(ctx)

	if len(req.Id) == 0 {
		xl.Error("empty username")
		return proto.ErrEmptyUsername
	}

	if len(req.Roles) == 0 {
		xl.Error("empty roles")
		return proto.ErrEmptyRoles
	}

	for _, r := range req.Roles {
		if r == proto.RoleAdmin {
			xl.Error("cannot create admin")
			return proto.ErrCannotCreateAdmin
		}
		if !r.IsValid() {
			xl.Error("invalid role")
			return proto.ErrInvalidRole
		}
	}

	user, err := dao.UserDao.Find(req.Id)
	if err != nil {
		xl.Errorf("userDao.Find(%s): %v", req.Id, err)
		return err
	}

	for _, r := range user.Roles {
		if r == proto.RoleAdmin {
			xl.Error("cannot update admin")
			return proto.ErrCannotUpdateAdmin
		}
	}

	err = dao.UserDao.Patch(req.Id, bson.M{
		"desc":  req.Desc,
		"roles": req.Roles,
	})
	if err != nil {
		xl.Errorf("userDao.Patch(%#v): %v", user, err)
		return err
	}
	dao.RoleCache.Set(user.Id, req.Roles)
	return nil
}

type UserPwdReq struct {
	OldPwd string `json:"old"`
	NewPwd string `json:"new"`
}

func (s *Service) PostUserPassword(
	ctx context.Context,
	req *UserPwdReq,
	env *restrpc.Env,
) error {
	xl := xlog.FromContextSafe(ctx)

	if len(req.OldPwd) == 0 || len(req.NewPwd) == 0 {
		xl.Error("empty password")
		return proto.ErrEmptyPassword
	}

	id := s.session.GetCurrentUserId(env.Req)
	user, err := dao.UserDao.Find(id)
	if err != nil {
		xl.Errorf("userDao.Find(%s): %v", id, err)
		return err
	}

	if user.Password != util.Sha1(req.OldPwd) {
		xl.Errorf("incorrect password")
		return proto.ErrIncorrectPwd
	}

	err = dao.UserDao.Patch(id, bson.M{
		"password": util.Sha1(req.NewPwd),
	})
	if err != nil {
		xl.Errorf("userDao.Patch(%#v): %v", user, err)
		return err
	}
	return nil
}

type UsersReq struct {
	Keyword string `json:"keyword"`
}

type UsersResp struct {
	Datas []*proto.User `json:"datas"`
}

func (_ *Service) GetUsers(
	ctx context.Context,
	req *UsersReq,
	env *restrpc.Env,
) (*UsersResp, error) {
	xl := xlog.FromContextSafe(ctx)

	users, err := dao.UserDao.Query(req.Keyword)
	if err != nil {
		xl.Errorf("userDao.Query: %v", err)
		return nil, err
	}

	resp := &UsersResp{
		Datas: users,
	}
	return resp, nil
}
