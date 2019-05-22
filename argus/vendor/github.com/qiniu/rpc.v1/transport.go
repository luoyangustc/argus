package rpc

import (
	"net"
	"net/http"
	"time"
)

const DefaultDailTimeout = time.Duration(5) * time.Second

var DefaultTransport = NewTransportTimeout(DefaultDailTimeout, 0)

func NewTransportTimeout(dial, resp time.Duration) http.RoundTripper {
	t := &http.Transport{ // DefaultTransport
		Proxy:               http.ProxyFromEnvironment,
		TLSHandshakeTimeout: 10 * time.Second,
	}
	t.Dial = (&net.Dialer{
		Timeout:   dial,
		KeepAlive: 30 * time.Second,
	}).Dial
	t.ResponseHeaderTimeout = resp
	return t
}

func NewTransportTimeoutWithConnsPool(dial, resp time.Duration, poolSize int) http.RoundTripper {

	t := &http.Transport{ // DefaultTransport
		Proxy:               http.ProxyFromEnvironment,
		TLSHandshakeTimeout: 10 * time.Second,
		MaxIdleConnsPerHost: poolSize,
	}
	t.Dial = (&net.Dialer{
		Timeout:   dial,
		KeepAlive: 30 * time.Second,
	}).Dial
	t.ResponseHeaderTimeout = resp
	return t
}
