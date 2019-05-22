package auth

import (
	"strconv"

	"github.com/qiniu/xlog.v1"
	"qbox.us/api/one/access"
	"qbox.us/qconf/qconfapi"
)

func AkSk(qCLI *qconfapi.Client, uid uint32) (string, string, error) {
	var ret access.AppInfo
	err := qCLI.Get(
		xlog.NewDummy(), &ret,
		"app:"+strconv.FormatUint(uint64(uid), 36)+":default",
		0)
	if err != nil {
		return "", "", err
	}
	return ret.Key, ret.Secret, nil
}
