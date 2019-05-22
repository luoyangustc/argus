package auth

import (
	"crypto/hmac"
	"crypto/sha1"
	"encoding/base64"
	"encoding/json"
	"net/url"
	"strconv"
	"strings"
	"time"
)

// ----------------------------------------------------------

// 根据空间(Bucket)的域名，以及文件的 key，获得 baseUrl。
// 如果空间是 public 的，那么通过 baseUrl 可以直接下载文件内容。
// 如果空间是 private 的，那么需要对 baseUrl 进行私有签名得到一个临时有效的 privateUrl 进行下载。
//
func MakeBaseUrl(domain, key string) (baseUrl string) {

	return "http://" + domain + "/" + url.QueryEscape(key)
}

// --------------------------------------------------------------------------------
// APIDOC: https://github.com/qbox/product/blob/master/kodo/up-uptoken.md
type PutPolicy struct {
	Scope               string `json:"scope"`
	Expires             int64  `json:"deadline"`             // 截止时间（以秒为单位）
	InsertOnly          uint16 `json:"insertOnly,omitempty"` // 若非0, 即使Scope为 Bucket:Key 的形式也是insert only
	DetectMime          uint8  `json:"detectMime,omitempty"` // 若非0, 则服务端根据内容自动确定 MimeType
	CallbackFetchKey    uint8  `json:"callbackFetchKey,omitempty"`
	FsizeMin            int64  `json:"fsizeMin,omitempty"`
	FsizeLimit          int64  `json:"fsizeLimit,omitempty"`
	MimeLimit           string `json:"mimeLimit,omitempty"`
	SaveKey             string `json:"saveKey,omitempty"`
	CallbackUrl         string `json:"callbackUrl,omitempty"`
	CallbackHost        string `json:"callbackHost,omitempty"`
	CallbackBody        string `json:"callbackBody,omitempty"`
	CallbackBodyType    string `json:"callbackBodyType,omitempty"`
	ReturnUrl           string `json:"returnUrl,omitempty"`
	ReturnBody          string `json:"returnBody,omitempty"`
	PersistentOps       string `json:"persistentOps,omitempty"`
	PersistentNotifyUrl string `json:"persistentNotifyUrl,omitempty"`
	PersistentPipeline  string `json:"persistentPipeline,omitempty"`
	AsyncOps            string `json:"asyncOps,omitempty"`
	EndUser             string `json:"endUser,omitempty"`
	Checksum            string `json:"checksum,omitempty"` // 格式：<HashName>:<HexHashValue>，目前支持 MD5/SHA1。
	DeleteAfterDays     int    `json:"deleteAfterDays,omitempty"`
	Transform           string `json:"transform,omitempty"`
	NotifyQueue         string `json:"notifyQueue,omitempty"`
	NotifyMessage       string `json:"notifyMessage,omitempty"`
}

func (p *PutPolicy) MakeUptoken(ak, sk string) string {

	var rr = *p
	if rr.Expires == 0 {
		rr.Expires = 3600
	}
	rr.Expires += int64(time.Now().Unix())
	b, _ := json.Marshal(&rr)

	return SignWithData(b, ak, sk)
}

func (p *PutPolicy) MakePrivateUrl(baseUrl, ak, sk string) string {
	var expires int64
	if p == nil || p.Expires == 0 {
		expires = 3600
	} else {
		expires = int64(p.Expires)
	}
	deadline := time.Now().Unix() + expires

	if strings.Contains(baseUrl, "?") {
		baseUrl += "&e="
	} else {
		baseUrl += "?e="
	}
	baseUrl += strconv.FormatInt(deadline, 10)

	token := Sign([]byte(baseUrl), ak, sk)

	return baseUrl + "&token=" + token
}

// ----------------------------------------------------------

func Sign(data []byte, ak, sk string) (token string) {

	h := hmac.New(sha1.New, []byte(sk))
	h.Write(data)

	sign := base64.URLEncoding.EncodeToString(h.Sum(nil))
	return ak + ":" + sign[:27]
}

func SignWithData(b []byte, ak, sk string) (token string) {

	blen := base64.URLEncoding.EncodedLen(len(b))

	key := ak
	nkey := len(key)
	ret := make([]byte, nkey+30+blen)

	base64.URLEncoding.Encode(ret[nkey+30:], b)

	h := hmac.New(sha1.New, []byte(sk))
	h.Write(ret[nkey+30:])
	digest := h.Sum(nil)

	copy(ret, key)
	ret[nkey] = ':'
	base64.URLEncoding.Encode(ret[nkey+1:], digest)
	ret[nkey+29] = ':'

	return string(ret)
}
