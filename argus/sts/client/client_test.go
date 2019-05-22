package client

import (
	"bytes"
	"context"
	"io/ioutil"
	"net/http"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/qiniu/rpc.v3"
)

func TestClientNewUrl(t *testing.T) {
	client := NewClient("127.0.0.1:5555", func() string { return "foo" }, nil)
	url, _ := client.NewURL(context.Background(), nil)
	assert.Equal(t, "sts://127.0.0.1:5555/v1/file/foo", url)
}

func TestClientGetURL(t *testing.T) {
	client := NewClient("127.0.0.1:5555", func() string { return "foo" }, nil)

	{
		uri, _ := client.GetURL(context.Background(), "sts://127.0.0.1:5555/v1/file/foo", nil, nil)
		assert.Equal(t, "sts://127.0.0.1:5555/v1/file/foo", uri)
	}
	{
		var length int64 = 10
		uri, _ := client.GetURL(context.Background(), "sts://127.0.0.1:5555/v1/file/foo", &length, nil)
		assert.Equal(t, "sts://127.0.0.1:5555/v1/file/foo", uri)
	}
	{
		uri, _ := client.GetURL(context.Background(), "http://qiniu.com", nil, nil)
		assert.Equal(t, "sts://127.0.0.1:5555/v1/fetch?uri=http%3A%2F%2Fqiniu.com", uri)
	}
	{
		var length int64 = 10
		uri, _ := client.GetURL(context.Background(), "http://qiniu.com", &length, nil)
		assert.Equal(t, "sts://127.0.0.1:5555/v1/fetch?uri=http%3A%2F%2Fqiniu.com&length=10", uri)
	}
	{
		var length int64 = 10
		uri, _ := client.GetURL(context.Background(), "http://qiniu.com?op=info", &length, nil)
		assert.Equal(t, "sts://127.0.0.1:5555/v1/fetch?uri=http%3A%2F%2Fqiniu.com%3Fop%3Dinfo&length=10", uri)
	}
	{
		var length int64 = 10
		var options = _OPTION_NONE | OPTION_SYNC
		uri, _ := client.GetURL(context.Background(), "http://qiniu.com?op=info", &length, &options)
		assert.Equal(t, "sts://127.0.0.1:5555/v1/fetch?uri=http%3A%2F%2Fqiniu.com%3Fop%3Dinfo&sync=true&length=10", uri)
	}
	{
		var length int64 = 10
		var options = _OPTION_NONE | OPTION_PROXY
		uri, _ := client.GetURL(context.Background(), "http://qiniu.com?op=info", &length, &options)
		assert.Equal(t, "sts://127.0.0.1:5555/v1/proxy?uri=http%3A%2F%2Fqiniu.com%3Fop%3Dinfo&length=10", uri)
	}
}

func TestOpenURI(t *testing.T) {
	ret, host, err := openURI(context.Background(), "sts://127.0.0.1:5555/v1/file/xxx")
	assert.NoError(t, err)
	assert.Equal(t, "http://127.0.0.1:5555/v1/open/xxx", ret)
	assert.Equal(t, "127.0.0.1:5555", host)
}

type mockRoundTripper struct {
	Func func(*http.Request) (*http.Response, error)
}

func (mock *mockRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	return mock.Func(req)
}

func TestClientSyncPost(t *testing.T) {

	var index = 0

	mock := &mockRoundTripper{
		Func: func(req *http.Request) (*http.Response, error) {
			switch index {
			case 0:
				index++
				return &http.Response{
					StatusCode: 200,
					Header: http.Header{
						"Content-Type": []string{"application/json"},
					},
					ContentLength: 11,
					Body:          ioutil.NopCloser(bytes.NewBufferString("{\"id\":\"xxxx\"}")),
				}, nil
			case 1:
				assert.Equal(t, "http://127.0.0.1:5555/v1/write/xxxx", req.URL.String())
				index++
				return &http.Response{
					StatusCode: 200,
					Body:       ioutil.NopCloser(bytes.NewBufferString("{}")),
				}, nil
			}

			return nil, nil
		},
	}

	client := &client{
		Client: rpc.NewClientWithTransport(mock),
		host:   "127.0.0.1:5555",
		newKey: func() string {
			return "xxx"
		},
	}

	uri, _ := client.NewURL(context.Background(), nil)
	done := func(err error) {} // NOTHING
	err := client.SyncPost(context.Background(), uri, 0, bytes.NewBufferString("test"), done)
	assert.NoError(t, err)
}
