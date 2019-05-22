package util

import (
	"encoding/base64"
	"net/http"
)

type BasicAuthTransport struct {
	authToken string
	transport http.RoundTripper
}

func NewBasicAuthTransport(username string, password string, trans http.RoundTripper) http.RoundTripper {
	if trans == nil {
		trans = http.DefaultTransport
	}

	return &BasicAuthTransport{
		authToken: BasicAuth(username, password),
		transport: trans,
	}
}

func (p *BasicAuthTransport) RoundTrip(req *http.Request) (resp *http.Response, err error) {
	req.Header.Set("Authorization", "Basic "+p.authToken)
	resp, err = p.transport.RoundTrip(req)
	return
}

func BasicAuth(username, password string) string {
	auth := username + ":" + password
	return base64.StdEncoding.EncodeToString([]byte(auth))
}

type AuthTokenTransport struct {
	token     string
	transport http.RoundTripper
}

func NewAuthTokenTransport(token string, trans http.RoundTripper) http.RoundTripper {
	if trans == nil {
		trans = http.DefaultTransport
	}
	return &AuthTokenTransport{
		token:     token,
		transport: trans,
	}
}

func (p *AuthTokenTransport) RoundTrip(req *http.Request) (resp *http.Response, err error) {
	req.Header.Set("X-Auth-Token", p.token)
	resp, err = p.transport.RoundTrip(req)
	return
}
