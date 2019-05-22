package proxy_auth

import (
	"net/http"
	"qbox.us/servend/account"
)

// ------------------------------------------------------------------------------------------

// Transport implements http.RoundTripper. When configured with a valid
// Config and Token it can be used to make authenticated HTTP requests.
//
//	c := NewClient(token, nil)
//	r, _, err := c.Get("http://example.org/url/requiring/auth")
//
type Transport struct {
	auth string

	// Transport is the HTTP transport to use when making requests.
	// It will default to http.DefaultTransport if nil.
	// (It should never be an oauth.Transport.)
	Transport http.RoundTripper
}

// RoundTrip executes a single HTTP transaction using the Transport's
// Token as authorization headers.
func (t *Transport) RoundTrip(req *http.Request) (resp *http.Response, err error) {
	req.Header.Set("Authorization", t.auth)
	return t.Transport.RoundTrip(req)
}

func (t *Transport) NestedObject() interface{} {
	return t.Transport
}

func NewTransport(user account.UserInfo, transport http.RoundTripper) *Transport {
	if transport == nil {
		transport = http.DefaultTransport
	}
	return &Transport{MakeAuth(user), transport}
}

func NewClient(user account.UserInfo, transport http.RoundTripper) *http.Client {
	t := NewTransport(user, transport)
	return &http.Client{Transport: t}
}

// ------------------------------------------------------------------------------------------
