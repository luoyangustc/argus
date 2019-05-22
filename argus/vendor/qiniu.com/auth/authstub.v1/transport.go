package authstub

import (
	"net/http"
	. "qiniu.com/auth/proto.v1"
)

// ---------------------------------------------------------------------------

type Transport struct {
	auth      string
	Transport http.RoundTripper
}

func (t *Transport) RoundTrip(req *http.Request) (resp *http.Response, err error) {

	req.Header.Set("Authorization", t.auth)
	return t.Transport.RoundTrip(req)
}

func (t *Transport) NestedObject() interface{} {

	return t.Transport
}

func NewTransport(user *SudoerInfo, transport http.RoundTripper) *Transport {

	if transport == nil {
		transport = http.DefaultTransport
	}
	return &Transport{Format(user), transport}
}

func NewClient(user *SudoerInfo, transport http.RoundTripper) *http.Client {

	t := NewTransport(user, transport)
	return &http.Client{Transport: t}
}

// ---------------------------------------------------------------------------
