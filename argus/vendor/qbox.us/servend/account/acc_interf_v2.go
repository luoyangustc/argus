package account

import (
	"net/http"

	"qiniu.com/auth/proto.v1"
)

// ---------------------------------------------------------------------------------------

const (
	// user type
	USER_TYPE_QBOX       = 0
	USER_TYPE_ADMIN      = proto.USER_TYPE_ADMIN
	USER_TYPE_VIP        = proto.USER_TYPE_VIP
	USER_TYPE_STDUSER    = proto.USER_TYPE_STDUSER
	USER_TYPE_STDUSER2   = proto.USER_TYPE_STDUSER2
	USER_TYPE_EXPUSER    = proto.USER_TYPE_EXPUSER
	USER_TYPE_PARENTUSER = proto.USER_TYPE_PARENTUSER
	USER_TYPE_OP         = proto.USER_TYPE_OP
	USER_TYPE_SUPPORT    = proto.USER_TYPE_SUPPORT
	USER_TYPE_CC         = proto.USER_TYPE_CC
	USER_TYPE_QCOS       = proto.USER_TYPE_QCOS
	USER_TYPE_PILI       = proto.USER_TYPE_PILI
	USER_TYPE_FUSION     = proto.USER_TYPE_FUSION
	USER_TYPE_DISABLED   = proto.USER_TYPE_DISABLED

	USER_TYPE_USERS            = proto.USER_TYPE_USERS
	USER_TYPE_SUDOERS          = proto.USER_TYPE_SUDOERS
	USER_TYPE_ENTERPRISE       = USER_TYPE_STDUSER
	USER_TYPE_ENTERPRISE_VUSER = USER_TYPE_STDUSER2
)

type UserInfo struct {
	Uid     uint32 `json:"uid"`
	Sudoer  uint32 `json:"suid,omitempty"`
	Utype   uint32 `json:"ut"`
	UtypeSu uint32 `json:"sut,omitempty"`
	Devid   uint32 `json:"dev,omitempty"`
	Appid   uint32 `json:"app,omitempty"`
	Expires uint32 `json:"e,omitempty"`
}

// ---------------------------------------------------------------------------------------

type AuthParser interface {
	ParseAuth(req *http.Request) (user UserInfo, err error)
}

// ---------------------------------------------------------------------------------------
