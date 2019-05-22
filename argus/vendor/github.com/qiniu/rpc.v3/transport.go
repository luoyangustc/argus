package rpc

import (
	"net"
	"net/http"
	"time"
)

// --------------------------------------------------------------------

type TransportConfig struct {
	DialTimeout           time.Duration
	ResponseHeaderTimeout time.Duration
	MaxIdleConnsPerHost   int
}

func NewTransport(cfg *TransportConfig) http.RoundTripper {

	t := &http.Transport{
		Proxy: http.ProxyFromEnvironment,
		ResponseHeaderTimeout: cfg.ResponseHeaderTimeout,
	}
	t.Dial = (&net.Dialer{
		Timeout:   cfg.DialTimeout,
		KeepAlive: 30 * time.Second,
	}).Dial
	return t
}

// --------------------------------------------------------------------
