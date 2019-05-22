package auth

import (
	"context"
	"net/http"
	"strings"

	"github.com/gorilla/securecookie"
	"github.com/gorilla/sessions"
	"github.com/qiniu/http/httputil.v1"
	xlog "github.com/qiniu/xlog.v1"
	"qiniu.com/argus/censor_private/dao"
	"qiniu.com/argus/censor_private/proto"
)

type Auth struct {
	ctx        context.Context
	cookieName string
	store      *sessions.CookieStore
}

func NewAuth(ctx context.Context, sessionTimeout int) *Auth {
	key := securecookie.GenerateRandomKey(32)
	store := sessions.NewCookieStore(key)
	store.MaxAge(sessionTimeout)
	return &Auth{
		ctx:        ctx,
		cookieName: "censor-private",
		store:      store,
	}
}

func (a *Auth) Refresh(req *http.Request, w http.ResponseWriter) {
	session, _ := a.store.Get(req, a.cookieName)
	_ = session.Save(req, w)
}

func (a *Auth) Login(req *http.Request, w http.ResponseWriter, user *proto.User) {
	session, _ := a.store.Get(req, a.cookieName)
	// clear previous session
	session.Values = make(map[interface{}]interface{})
	session.Values["id"] = user.Id
	_ = session.Save(req, w)
}

func (a *Auth) Logout(req *http.Request, w http.ResponseWriter) {
	session, _ := a.store.Get(req, a.cookieName)
	session.Values = make(map[interface{}]interface{})
	_ = session.Save(req, w)
}

func (a *Auth) Validate(req *http.Request) error {
	xl := xlog.FromContextSafe(a.ctx)
	path := strings.ToLower(req.URL.Path)
	userID := a.GetCurrentUserId(req)
	roles := []proto.Role{}

	var isAdmin, canCensor, canManageSet, authenticated = false, false, false, false

	if len(userID) != 0 {
		authenticated = true

		roles = dao.RoleCache.Get(userID)
		if roles == nil {
			user, err := dao.UserDao.Find(userID)
			if err != nil {
				xl.Errorf("dao.UserDao.Find(%s): %v", userID, err)
				authenticated = false
			} else {
				roles = user.Roles
				dao.RoleCache.Set(userID, roles)
			}
		}
	}

	for _, item := range roles {
		switch item {
		case proto.RoleAdmin:
			isAdmin = true
		case proto.RoleCensor:
			canCensor = true
		case proto.RoleManageSet:
			canManageSet = true
		default:
		}
	}

	var canAccess bool
	switch {
	case strings.HasPrefix(path, "/v1/censor"):
		// 审核需要审核权限或admin
		canAccess = isAdmin || canCensor
	case path == "/v1/sets":
		// 获取监控列表需要监控、审核或admin权限
		canAccess = isAdmin || canManageSet || canCensor
	case strings.HasPrefix(path, "/v1/set"):
		// 其他监控操作需要监控权限或admin
		canAccess = isAdmin || canManageSet
	case path == "/v1/user/password":
		// 已登录用户均可修改其密码
		canAccess = authenticated
	case strings.HasPrefix(path, "/v1/user"):
		// 用户管理只能admin操作
		canAccess = isAdmin
	default:
		// 其他登录等操作，无需权限
		canAccess = true
	}

	if canAccess {
		return nil
	} else if !authenticated {
		return proto.ErrUnauthorized
	} else {
		return proto.ErrForbidden
	}
}

func (a *Auth) Handler(
	w http.ResponseWriter, req *http.Request, f func(http.ResponseWriter, *http.Request)) {
	// validate request based on session and authorization
	if err := a.Validate(req); err != nil {
		httputil.Error(w, err)
		return
	}

	// refresh session
	// NOTE: must called before propagate request
	a.Refresh(req, w)

	// propagate request
	f(w, req)
}

func (a *Auth) GetCurrentUserId(req *http.Request) string {
	session, _ := a.store.Get(req, a.cookieName)
	id, _ := session.Values["id"].(string)
	return id
}
