package auth

import (
	"crypto/hmac"
	"crypto/sha1"
	"encoding/base64"
	"net/url"
	"os"
	"strconv"
	"strings"
	"time"
)

type AccessInfo struct {
	Key    string `json:"key"`
	Secret string `json:"secret"`
}

func SignUptoken(user AccessInfo, bucket string) string {
	policy := PutPolicy{
		Scope: bucket,
	}
	return policy.MakeUptoken(user.Key, user.Secret)
}

func SignUpTokenWithPutPolicy(user AccessInfo, putPolicy PutPolicy) string {
	return putPolicy.MakeUptoken(user.Key, user.Secret)
}

func GetPrivateUrl(baseUrl, ak, sk string) string {
	if ak == "" && sk == "" {
		if os.Getenv("TEST_ENV") != "private" {
			println("ak & sk can't be null")
		}
	}
	policy := PutPolicy{}

	return policy.MakePrivateUrl(baseUrl, ak, sk)
}

// QBox Authorization
// DOC: https://github.com/qbox/bs-apidoc/blob/master/apidoc/v6/auths/Qbox.md
func SignQboxToken(user AccessInfo, uri, body string) string {
	u, err := url.Parse(uri)
	if err != nil {
		println("Parse url failed, url = %d", uri)
	}

	data := u.Path

	if u.RawQuery != "" {
		data += "?" + u.RawQuery
	}
	data += "\n"

	if body != "" {
		data += body
	}

	h := hmac.New(sha1.New, []byte(user.Secret))
	h.Write([]byte(data))
	sign := base64.URLEncoding.EncodeToString(h.Sum(nil))

	return user.Key + ":" + sign
}

// QBox Authorization
// DOC: https://github.com/qbox/bs-apidoc/blob/master/apidoc/v6/auths/Qbox.md
func SignQboxAdminToken(user AccessInfo, uri, body, contentType, suInfo string) string {
	u, err := url.Parse(uri)
	if err != nil {
		println("Parse url failed, url = %d", uri)
	}

	data := u.Path

	if u.RawQuery != "" {
		data += "?" + u.RawQuery
	}

	data += "\nAuthorization: QBoxAdmin " + suInfo + "\n\n"

	if contentType == "application/x-www-form-urlencoded" && body != "" {
		data += body
	}

	h := hmac.New(sha1.New, []byte(user.Secret))
	h.Write([]byte(data))
	sign := base64.URLEncoding.EncodeToString(h.Sum(nil))

	return suInfo + ":" + user.Key + ":" + sign
}

func SignDownloadURL(user AccessInfo, url string) string {
	var expires int64 = 3600 // 一小时内有效
	deadline := time.Now().Unix() + expires
	if strings.Contains(url, "?") {
		url += "&e="
	} else {
		url += "?e="
	}

	url += strconv.FormatInt(deadline, 10)
	token := Sign([]byte(url), user.Key, user.Secret)

	url += "&token=" + token
	return url
}
