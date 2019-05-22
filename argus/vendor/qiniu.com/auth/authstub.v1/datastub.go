package authstub

import (
	"strings"

	. "qiniu.com/auth/proto.v1"
)

func FormatStubData(user *SudoerInfo, data string) string {

	return "QiniuStubData " + FormatToken(user) + ":" + data
}

func ParseStubData(auth string) (user SudoerInfo, data string, err error) {

	if strings.HasPrefix(auth, "QiniuStubData ") {
		auth = auth[14:]
		if idx := strings.Index(auth, ":"); idx != -1 {
			user, err = ParseToken(auth[:idx])
			data = auth[idx+1:]
			return
		}
	}
	err = ErrBadToken
	return
}
