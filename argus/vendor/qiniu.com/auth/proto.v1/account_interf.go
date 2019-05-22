package proto

import (
	. "context"
)

// --------------------------------------------------------------------

const (
	USER_TYPE_ADMIN      = 0x0001
	USER_TYPE_VIP        = 0x0002
	USER_TYPE_STDUSER    = 0x0004
	USER_TYPE_STDUSER2   = 0x0008
	USER_TYPE_EXPUSER    = 0x0010
	USER_TYPE_PARENTUSER = 0x0020
	USER_TYPE_OP         = 0x0040
	USER_TYPE_SUPPORT    = 0x0080
	USER_TYPE_CC         = 0x0100
	USER_TYPE_QCOS       = 0x0200
	USER_TYPE_FUSION     = 0x0400
	USER_TYPE_PILI       = 0x0800
	USER_TYPE_DISABLED   = 0x8000

	USER_TYPE_USERS   = USER_TYPE_STDUSER | USER_TYPE_STDUSER2 | USER_TYPE_EXPUSER
	USER_TYPE_SUDOERS = USER_TYPE_ADMIN | USER_TYPE_OP | USER_TYPE_SUPPORT
)

type UserInfo struct {
	Uid    uint32 `json:"uid"`
	Utype  uint32 `json:"ut"`
	Appid  uint64 `json:"app,omitempty"`
	Access string `json:"ak,omitempty"`

	EndUser string `json:"eu,omitempty"`
}

type SudoerInfo struct {
	UserInfo
	Sudoer  uint32 `json:"suid,omitempty"`
	UtypeSu uint32 `json:"sut,omitempty"`
}

// --------------------------------------------------------------------

type AccessInfo struct {
	Secret []byte `bson:"secret"`
	Appid  uint64 `bson:"appId,omitempty"`
	Uid    uint32 `bson:"uid"`
}

type Interface interface {
	GetAccessInfo(ctx Context, accessKey string) (ret AccessInfo, err error)
	GetUtype(ctx Context, uid uint32) (utype uint32, err error)
}

// --------------------------------------------------------------------
