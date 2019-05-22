package qboxmac

import (
	"encoding/base64"
	"net/http"

	. "github.com/qiniu/api/conf"
)

// ---------------------------------------------------------------------------------------

type Mac struct {
	AccessKey string
	SecretKey []byte
}

type Transport struct {
	mac       Mac
	Transport http.RoundTripper
}

func (t *Transport) RoundTrip(req *http.Request) (resp *http.Response, err error) {

	sign, err := SignRequest(t.mac.SecretKey, req)
	if err != nil {
		return
	}

	auth := "QBox " + t.mac.AccessKey + ":" + base64.URLEncoding.EncodeToString(sign)
	req.Header.Set("Authorization", auth)
	return t.Transport.RoundTrip(req)
}

func (t *Transport) NestedObject() interface{} {

	return t.Transport
}

func NewTransport(mac *Mac, transport http.RoundTripper) *Transport {

	if transport == nil {
		transport = http.DefaultTransport
	}
	t := &Transport{Transport: transport}
	if mac == nil {
		t.mac.AccessKey = ACCESS_KEY
		t.mac.SecretKey = []byte(SECRET_KEY)
	} else {
		t.mac = *mac
	}
	return t
}

func NewClient(mac *Mac, transport http.RoundTripper) *http.Client {

	t := NewTransport(mac, transport)
	return &http.Client{Transport: t}
}

// ---------------------------------------------------------------------------------------

type AdminTransport struct {
	mac       Mac
	suInfo    string
	Transport http.RoundTripper
}

func (t *AdminTransport) RoundTrip(req *http.Request) (resp *http.Response, err error) {

	sign, err := SignAdminRequest(t.mac.SecretKey, req, t.suInfo)
	if err != nil {
		return
	}

	auth := "QBoxAdmin " + t.suInfo + ":" + t.mac.AccessKey + ":" + base64.URLEncoding.EncodeToString(sign)
	req.Header.Set("Authorization", auth)
	return t.Transport.RoundTrip(req)
}

func (t *AdminTransport) NestedObject() interface{} {

	return t.Transport
}

func NewAdminTransport(mac *Mac, suInfo string, transport http.RoundTripper) *AdminTransport {

	if transport == nil {
		transport = http.DefaultTransport
	}
	return &AdminTransport{Transport: transport, mac: *mac, suInfo: suInfo}
}

// ---------------------------------------------------------------------------------------

