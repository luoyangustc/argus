package uri

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"net/http"
	"strconv"
	"strings"

	"github.com/pkg/errors"
)

type httpHandler struct {
	ioHost string
	client *http.Client
}

func (h *httpHandler) Get(ctx context.Context, args Request, opts ...GetOption,
) (resp *Response, err error) {

	for _, opt := range opts {
		opt(&args)
	}

	url := args.URI
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return nil, errors.Wrap(err, "getHTTP http.NewRequest")
	}
	// 这段逻辑为 withHTTPFromIOHandler 服务，用于直接从源站下载
	if h.ioHost != "" {
		req.URL.Host = h.ioHost
	}

	if args.beginOff != nil {
		FormatRangeRequest(req.Header, args)
	}

	req = req.WithContext(ctx)
	_resp, err := h.client.Do(req)
	if err != nil {
		return nil, errors.Wrap(err, "getHTTP client.Do")
	}
	if _resp.StatusCode/100 != 2 {
		defer _resp.Body.Close()
		return nil, responseError(_resp)
	}
	return transResp(_resp), err
}

func (h *httpHandler) Names() []string {
	return []string{"http", "https"}
}

func transResp(resp *http.Response) *Response {
	size, _ := strconv.ParseInt(resp.Header.Get("content-length"), 10, 64)
	return &Response{
		Header: resp.Header,
		Body:   resp.Body,
		Size:   size,
	}
}

func withHTTPFromIOHandler(ioHost string) Handler {
	ioHost = strings.TrimPrefix(ioHost, "http://")
	ioHost = strings.TrimPrefix(ioHost, "https://")
	return &httpHandler{
		ioHost: ioHost,
		client: http.DefaultClient,
	}
}

func WithHTTPHandler() Handler {
	return &httpHandler{
		client: http.DefaultClient,
	}
}

func WithHTTPPublicHandler() Handler {
	return &httpHandler{
		client: &http.Client{Transport: OnlyPublicIPHTTPClient()},
	}
}

func WithSimpleHTTPHandler(client *http.Client) Handler {
	return &httpHandler{
		client: client,
	}
}

func WithCertHTTPSHandler(caCert, clientCert, clientKey []byte) Handler {
	var (
		pool  *x509.CertPool
		err   error
		certs []tls.Certificate
	)
	pool, err = x509.SystemCertPool()
	if err != nil {
		pool = x509.NewCertPool()
	}
	if len(caCert) > 0 {
		pool.AppendCertsFromPEM(caCert)
	}
	certs = make([]tls.Certificate, 0)
	if len(clientCert) > 0 && len(clientKey) > 0 {
		if cert, err := tls.X509KeyPair(clientCert, clientKey); err == nil {
			certs = append(certs, cert)
		}
	}

	tr := &http.Transport{
		TLSClientConfig: &tls.Config{
			RootCAs:            pool,
			Certificates:       certs,
			InsecureSkipVerify: true,
		},
	}
	return &httpHandler{
		client: &http.Client{Transport: tr},
	}
}
