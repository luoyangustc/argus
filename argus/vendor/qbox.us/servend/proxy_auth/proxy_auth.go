package proxy_auth

import (
	"strconv"
	"strings"
	"syscall"

	"net/http"
	"net/url"

	"qbox.us/servend/account"

	"github.com/qiniu/errors"
	"github.com/qiniu/http/formutil.v1"
	"github.com/qiniu/http/httputil.v1"
)

/*
  Authorization: QiniuProxy uid=$Uid&suid=$Sudoer&ut=$Utype&sut=$UtypeSu&dev=$Devid&app=$Appid&e=$Expires
*/

var ErrBadToken = httputil.NewError(401, "bad token")

func appendUint32(form []byte, k string, v uint32) []byte {

	str := strconv.FormatUint(uint64(v), 10)
	form = append(form, k...)
	return append(form, str...)
}

func MakeAuth(user account.UserInfo) string {

	form := make([]byte, 0, 64)
	form = append(form, "QiniuProxy "...)

	form = appendUint32(form, "uid=", user.Uid)
	form = appendUint32(form, "&ut=", user.Utype)
	if user.Sudoer != 0 {
		form = appendUint32(form, "&suid=", user.Sudoer)
	}
	if user.UtypeSu != 0 {
		form = appendUint32(form, "&sut=", user.UtypeSu)
	}
	if user.Devid != 0 {
		form = appendUint32(form, "&dev=", user.Devid)
	}
	if user.Appid != 0 {
		form = appendUint32(form, "&app=", user.Appid)
	}
	if user.Expires != 0 {
		form = appendUint32(form, "&e=", user.Expires)
	}
	return string(form)
}

func ParseAuth(req *http.Request) (user account.UserInfo, err error) {

	if auth1, ok := req.Header["Authorization"]; ok {
		auth := auth1[0]
		if strings.HasPrefix(auth, "QiniuProxy ") {
			m, err1 := url.ParseQuery(auth[11:])
			if err1 != nil {
				err = errors.Info(ErrBadToken, "proxy.GetAuth: ParseQuery failed", auth).Detail(err1)
				return
			}
			err = formutil.Parse(&user, m)
			if err != nil {
				err = errors.Info(ErrBadToken, "proxy.GetAuth: formutil.Parse failed", auth).Detail(err)
			}
			return
		}
	}
	err = syscall.EACCES
	return
}

// ---------------------------------------------------------------------------

type parser struct{}

func (p parser) ParseAuth(req *http.Request) (user account.UserInfo, err error) {
	return ParseAuth(req)
}

var Parser parser

// ---------------------------------------------------------------------------
