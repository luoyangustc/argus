package auth

import (
	"crypto/hmac"
	"crypto/sha1"
	"encoding/base64"
	"fmt"
	"net/url"
	"strconv"
	"time"
)

type QUserAuthArgs struct {
	User        AccessInfo
	Method      string
	URL         string
	Body        string
	ContentType string
	UID         uint32
	Collname    string
}

// DOC: https://github.com/qbox/bs-apidoc/blob/master/apidoc/v6/auths/QUser.md
func SignQUserAuthToken(args QUserAuthArgs) string {
	str := args.Method + " "
	u, err := url.Parse(args.URL)
	if err != nil {
		panic(err)
	}
	str += u.Path
	if u.RawQuery != "" {
		str += "?" + u.RawQuery
	}
	str += "\n"
	str += "Host: " + u.Host + "\n"
	str += "Content-Type: " + args.ContentType + "\n"

	quserAK := args.User.Key + "/" + fmt.Sprintf("%v", time.Now().Add(24*time.Hour).Unix())
	quserAK += "/euid=" + strconv.FormatInt(int64(args.UID), 10) + "&coll=" + args.Collname

	str += "Authorization: QUser " + quserAK + "\n\n"

	if args.Body != "" && args.ContentType != "application/octet-stream" {
		str += args.Body
	}

	h := hmac.New(sha1.New, []byte(args.User.Secret))
	h.Write([]byte(quserAK))
	QUserSK := base64.URLEncoding.EncodeToString(h.Sum(nil))

	q := hmac.New(sha1.New, []byte(QUserSK))
	q.Write([]byte(str))
	QUserSign := base64.URLEncoding.EncodeToString(q.Sum(nil))

	return quserAK + ":" + QUserSign
}
