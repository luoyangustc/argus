package oauth

import (
	"net/http"
	"strings"
	"syscall"
)

// ------------------------------------------------------------------------------------------

// Transport implements http.RoundTripper. When configured with a valid
// Config and Token it can be used to make authenticated HTTP requests.
//
//	c := NewClient(token, nil)
//	r, _, err := c.Get("http://example.org/url/requiring/auth")
//
type Transport struct {
	token string // access token

	// Transport is the HTTP transport to use when making requests.
	// It will default to http.DefaultTransport if nil.
	// (It should never be an oauth.Transport.)
	transport http.RoundTripper
}

// RoundTrip executes a single HTTP transaction using the Transport's
// Token as authorization headers.
func (t *Transport) RoundTrip(req *http.Request) (resp *http.Response, err error) {
	req.Header.Set("Authorization", "Bearer "+t.token)
	return t.transport.RoundTrip(req)
}

func (t *Transport) NestedObject() interface{} {
	return t.transport
}

func NewTransport(token string, transport http.RoundTripper) *Transport {
	if transport == nil {
		transport = http.DefaultTransport
	}
	return &Transport{token, transport}
}

func NewClient(token string, transport http.RoundTripper) *http.Client {
	t := NewTransport(token, transport)
	return &http.Client{Transport: t}
}

func GetAccessToken(req *http.Request) (token string, err error) {
	if auth1, ok := req.Header["Authorization"]; ok {
		auth := auth1[0]
		if strings.HasPrefix(auth, "Bearer ") {
			token = auth[7:]
			return
		}
	}
	err = syscall.EACCES
	return
}

// ------------------------------------------------------------------------------------------

const (
	BearerToken = 1
	QBoxToken   = 2
)

func GetAccessTokenEx(req *http.Request) (typ int, token string, err error) {
	if auth1, ok := req.Header["Authorization"]; ok {
		auth := auth1[0]
		if strings.HasPrefix(auth, "Bearer ") {
			return BearerToken, auth[7:], nil
		} else if strings.HasPrefix(auth, "QBox ") {
			return QBoxToken, auth[5:], nil
		}
	}
	err = syscall.EACCES
	return
}

// ------------------------------------------------------------------------------------------
