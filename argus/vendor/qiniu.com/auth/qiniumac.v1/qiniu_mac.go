package qiniumac

import (
	"crypto/hmac"
	"crypto/sha1"
	"io"
	"net/http"
	"sort"

	"github.com/qiniu/bytes/seekable"
)

const qiniuHeaderPrefix = "X-Qiniu-"

// ---------------------------------------------------------------------------------------

func incBody(req *http.Request, ctType string) bool {

	return req.ContentLength != 0 && req.Body != nil && ctType != "" && ctType != "application/octet-stream"
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

func SignRequest(sk []byte, req *http.Request) ([]byte, error) {

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
		s2, err2 := seekable.New(req)
		if err2 != nil {
			return nil, err2
		}
		h.Write(s2.Bytes())
	}

	return h.Sum(nil), nil
}

func SignAdminRequest(sk []byte, req *http.Request, su string) ([]byte, error) {

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
	io.WriteString(h, "\nAuthorization: QiniuAdmin "+su)

	signQiniuHeaderValues(req.Header, h)

	io.WriteString(h, "\n\n")

	if incBody(req, ctType) {
		s2, err2 := seekable.New(req)
		if err2 != nil {
			return nil, err2
		}
		h.Write(s2.Bytes())
	}

	return h.Sum(nil), nil
}

// ---------------------------------------------------------------------------------------

type RequestSigner struct {
}

var (
	DefaultRequestSigner RequestSigner
)

func (p RequestSigner) Sign(sk []byte, req *http.Request) ([]byte, error) {

	return SignRequest(sk, req)
}

func (p RequestSigner) SignAdmin(sk []byte, req *http.Request, su string) ([]byte, error) {

	return SignAdminRequest(sk, req, su)
}

// ---------------------------------------------------------------------------------------

type sortByHeaderKey []string

func (p sortByHeaderKey) Len() int           { return len(p) }
func (p sortByHeaderKey) Less(i, j int) bool { return p[i] < p[j] }
func (p sortByHeaderKey) Swap(i, j int)      { p[i], p[j] = p[j], p[i] }

// ---------------------------------------------------------------------------------------
