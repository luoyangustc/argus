package auth

import (
	"crypto/hmac"
	"crypto/sha1"
	"encoding/base64"
	"io"
	"net/http"
	"net/url"
	"sort"

	"qiniu.com/argus/test/lib/qnhttp"
)

type QiniuAuthArgs struct {
	User        AccessInfo
	Method      string
	URL         string
	Body        string
	ContentType string
	SuInfo      string
	Host        string
}

// DOC: https://github.com/qbox/bs-apidoc/blob/master/apidoc/v6/auths/Qiniu.md
func SignQiniuAuthToken(args QiniuAuthArgs) string {
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
	if args.Host != "" {
		str += "Host: " + args.Host + "\n"
	} else {
		str += "Host: " + u.Host + "\n"
	}
	str += "Content-Type: " + args.ContentType + "\n\n"
	if args.Body != "" && args.ContentType != "application/octet-stream" {
		str += args.Body
	}

	h := hmac.New(sha1.New, []byte(args.User.Secret))
	h.Write([]byte(str))
	signed := base64.URLEncoding.EncodeToString(h.Sum(nil))

	return args.User.Key + ":" + signed
}

//https://github.com/qbox/base/blob/master/qiniu/src/qiniu.com/auth/qiniumac.v1/README.md#qiniuadmin-authorization
func SignQiniuAdminToken(args QiniuAuthArgs) string {
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
	if args.Host != "" {
		str += "Host: " + args.Host + "\n"
	} else {
		str += "Host: " + u.Host + "\n"
	}
	str += "Content-Type: " + args.ContentType + "\n"
	str += "Authorization: QiniuAdmin " + args.SuInfo + "\n\n"
	println(str)

	if args.Body != "" && args.ContentType != "application/octet-stream" {
		str += args.Body
	}

	h := hmac.New(sha1.New, []byte(args.User.Secret))
	h.Write([]byte(str))
	signed := base64.URLEncoding.EncodeToString(h.Sum(nil))

	return args.SuInfo + ":" + args.User.Key + ":" + signed
}

//https://github.com/qbox/base/blob/master/qiniu/src/qiniu.com/auth/qiniumac.v1/README.md#qiniuadmin-authorization
func SignQiniuAdminRequest(user AccessInfo, req *qnhttp.Session, su string) {
	h := hmac.New(sha1.New, []byte(user.Secret))

	u, err := url.Parse(req.URL)
	println(req.URL)
	if err != nil {
		panic(err)
	}

	data := req.Method + " " + u.Path
	if u.RawQuery != "" {
		data += "?" + u.RawQuery
	}
	io.WriteString(h, data+"\nHost: "+u.Host)

	ctType := req.Header.Get("Content-Type")
	if ctType != "" {
		io.WriteString(h, "\nContent-Type: "+ctType)
	}
	io.WriteString(h, "\nAuthorization: QiniuAdmin "+su)

	signQiniuHeaderValues(*req.Header, h)

	io.WriteString(h, "\n\n")

	if ctType != "application/octet-stream" && req.Body != "" {
		h.Write([]byte(req.Body))
	}

	auth := "QiniuAdmin " + su + ":" + user.Key + ":" + base64.URLEncoding.EncodeToString(h.Sum(nil))
	req.Header.Set("Authorization", auth)
}

//-------------------------------------------

const qiniuHeaderPrefix = "X-Qiniu-"

type Mac struct {
	AccessKey string
	SecretKey []byte
}

type QiniuTransport struct {
	mac       Mac
	Transport http.RoundTripper
}

func (t *QiniuTransport) RoundTrip(req *http.Request) (resp *http.Response, err error) {
	sign, err := t.SignRequest(t.mac.SecretKey, req)
	if err != nil {
		return
	}

	auth := "Qiniu " + t.mac.AccessKey + ":" + base64.URLEncoding.EncodeToString(sign)
	req.Header.Set("Authorization", auth)
	return t.Transport.RoundTrip(req)
}

func NewQiniuTransport(mac *Mac, transport http.RoundTripper) *QiniuTransport {

	if transport == nil {
		transport = http.DefaultTransport
	}
	t := &QiniuTransport{Transport: transport}
	t.mac = *mac
	return t
}

func (t *QiniuTransport) SignRequest(sk []byte, req *http.Request) ([]byte, error) {

	h := hmac.New(sha1.New, sk)

	u := req.URL
	data := req.Method + " " + u.Path
	if u.RawQuery != "" {
		data += "?" + u.RawQuery
	}
	io.WriteString(h, data+"\nHost: "+req.Host)

	ctType := req.Header.Get("Content-Type")
	if ctType != "" {
		io.WriteString(h, "\nContent-Type: "+ctType)
	}
	signQiniuHeaderValues(req.Header, h)

	io.WriteString(h, "\n\n")

	if incBody(req, ctType) {
		s2, err2 := New(req)
		if err2 != nil {
			return nil, err2
		}
		h.Write(s2.Bytes())
	}

	return h.Sum(nil), nil
}

func signQiniuHeaderValues(header http.Header, w io.Writer) {
	var keys []string
	for key, _ := range header {
		if len(key) > len(qiniuHeaderPrefix) && key[:len(qiniuHeaderPrefix)] == qiniuHeaderPrefix {
			keys = append(keys, key)
		}
	}
	if len(keys) == 0 {
		return
	}

	if len(keys) > 1 {
		sort.Sort(sortByHeaderKey(keys))
	}
	for _, key := range keys {
		io.WriteString(w, "\n"+key+": "+header.Get(key))
	}
}

type sortByHeaderKey []string

func (p sortByHeaderKey) Len() int           { return len(p) }
func (p sortByHeaderKey) Less(i, j int) bool { return p[i] < p[j] }
func (p sortByHeaderKey) Swap(i, j int)      { p[i], p[j] = p[j], p[i] }

func incBody(req *http.Request, ctType string) bool {

	return req.ContentLength != 0 && req.Body != nil && ctType != "" && ctType != "application/octet-stream"
}
