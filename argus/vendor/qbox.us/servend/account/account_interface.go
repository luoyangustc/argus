package account

import (
	"net/http"

	"github.com/qiniu/log.v1"
	"qbox.us/api"
	"qbox.us/errors"
	"qbox.us/servend/oauth"
)

// ---------------------------------------------------------------------------------------

const (
	APP_INVALID       = 0
	APP_QBOX_INTERNAL = 1
	APP_QBOX_IPHONE   = 2
	APP_QBOX_ANDROID  = 3
	APP_QBOX_WINDOWS  = 4
	APP_QBOX_WEB      = 5
	APP_QBOX_MOBIWEB  = 6
	APP_QBOX_IPAD     = 7
	APP_CAMERA360     = 1001
	APP_WEICO         = 1002
)

// ---------------------------------------------------------------------------------------
// [[DEPRECATED]]

type Interface interface {
	ParseAccessToken(token string) (user UserInfo, err error)
	MakeAccessToken(user UserInfo) string
}

// 如果没有 Authorization 头，或者 Authorization 头不是以 `Bearer `开头，返回 syscall.EACCES
// 如果 Token 解码失败，返回 "qbox.us/account/encryptor".DecodeError 的一个实例
// 如果 Token 已经过期，返回 "qbox.us/account".ErrTokenExpired
func GetAuth(acc Interface, req *http.Request) (user UserInfo, err error) {
	token, err := oauth.GetAccessToken(req)
	if err != nil {
		return
	}
	return acc.ParseAccessToken(token)
}

func CheckAuth(acc Interface, req *http.Request, authTypes uint32) (err error) {
	user, err := GetAuth(acc, req)
	if err != nil {
		return
	}
	if (user.Utype & authTypes) == 0 {
		return api.EBadToken
	}
	return nil
}

func CheckOperator(acc Interface, req *http.Request) (err error) {
	return CheckAuth(acc, req, USER_TYPE_OP|USER_TYPE_ADMIN)
}

// ---------------------------------------------------------------------------------------
// [[DEPRECATED]]

type InterfaceEx interface {
	Interface
	DigestAuth(token string, req *http.Request) (user UserInfo, err error)
	DigestAuthEx(tempToken string) (user UserInfo, data string, err error)
	GetSecret(key string) (secret []byte, ok bool)
}

func GetAuthExt(acc InterfaceEx, req *http.Request) (user UserInfo, err error) {
	typ, token, err := oauth.GetAccessTokenEx(req)
	log.Debug("GetAuthExt:", typ, token, err)
	if err != nil {
		err = errors.Info(api.EBadToken, "GetAuthExt").Detail(err)
		return
	}
	switch typ {
	case oauth.QBoxToken:
		user, err = acc.DigestAuth(token, req)
	case oauth.BearerToken:
		user, err = acc.ParseAccessToken(token)
	default:
		err = errors.New("unexcepted")
	}
	if err != nil {
		err = errors.Info(api.EBadToken, "GetAuthExt").Detail(err)
		log.Error(errors.Detail(err))
	}
	return
}

func CheckAuthExt(acc InterfaceEx, req *http.Request, authTypes uint32) (err error) {
	user, err := GetAuthExt(acc, req)
	if err != nil {
		return
	}
	if (user.Utype & authTypes) == 0 {
		return api.EBadToken
	}
	return nil
}

func CheckGetAuthExt(acc InterfaceEx, req *http.Request, authTypes uint32) (user UserInfo, err error) {
	user, err = GetAuthExt(acc, req)
	if err != nil {
		return
	}
	if (user.Utype & authTypes) == 0 {
		err = api.EBadToken
		return
	}
	return
}

func CheckEnterpriseExt(acc InterfaceEx, req *http.Request) (err error) {
	return CheckAuthExt(acc, req, USER_TYPE_ENTERPRISE|USER_TYPE_ENTERPRISE_VUSER|USER_TYPE_ADMIN)
}

// ---------------------------------------------------------------------------------------
