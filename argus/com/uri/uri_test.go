package uri

import (
	"context"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"os"
	"strconv"
	"testing"
	"time"

	"github.com/stretchr/testify.v2/assert"
)

func TestClient_GetBadUri(t *testing.T) {
	assert := assert.New(t)
	c := New()
	ctx, cancel := context.WithTimeout(context.Background(), time.Second*5)
	defer cancel()
	_, err := c.Get(ctx, Request{URI: "xxx"})
	assert.Equal(err, ErrNotSupported)
}

func TestClient_GetHTTP(t *testing.T) {
	if os.Getenv("TRAVIS") != "" {
		t.SkipNow()
	}
	assert := assert.New(t)
	c := New(WithHTTPHandler())
	ctx, cancel := context.WithTimeout(context.Background(), time.Second*5)
	defer cancel()
	resp, err := c.Get(ctx, Request{URI: "https://www.baidu.com/"})
	assert.Nil(err)
	defer resp.Body.Close()
	buf, err := ioutil.ReadAll(resp.Body)
	assert.Nil(err)
	assert.Contains(string(buf), "baidu")
}

func TestClient_GetHTTPFromIO(t *testing.T) {
	if os.Getenv("TRAVIS") != "" {
		t.SkipNow()
	}
	assert := assert.New(t)
	c := New(withHTTPFromIOHandler("http://iovip.qbox.me"))
	{
		ctx, cancel := context.WithTimeout(context.Background(), time.Second*5)
		defer cancel()
		resp, err := c.Get(ctx, Request{URI: "https://www.baidu.com/"})
		assert.NotNil(err)
		assert.Nil(resp)
	}
	{
		ctx, cancel := context.WithTimeout(context.Background(), time.Second*5)
		defer cancel()
		resp, err := c.Get(ctx, Request{URI: "http://q.hi-hi.cn/1.png"})
		assert.Nil(err)
		defer resp.Body.Close()
	}
}

func TestClient_GetQiniu(t *testing.T) {
	ak := os.Getenv("AK")
	sk := os.Getenv("SK")
	if ak == "" {
		t.SkipNow()
		return
	}
	assert := assert.New(t)
	c := New(WithUserAkSk(ak, sk, "http://rs.qbox.me"))
	ctx, cancel := context.WithTimeout(context.Background(), time.Second*5)
	defer cancel()
	resp, err := c.Get(ctx, Request{URI: "qiniu://1380585377@z0/test/1.png"})
	assert.Nil(err)
	defer resp.Body.Close()
	n, _ := strconv.Atoi(resp.Header.Get("content-length"))
	assert.True(n > 10000)
	assert.True(int64(n) == resp.Size)
	assert.Nil(err)

	ctx, cancel = context.WithTimeout(context.Background(), time.Microsecond)
	defer cancel()
	resp, err = c.Get(ctx, Request{URI: "qiniu://1380585377@z0/test/1.png"})
	assert.Contains(err.Error(), "context deadline exceeded")
}

func TestClient_GetQiniuAdmin(t *testing.T) {
	ak := os.Getenv("ADMINAK")
	sk := os.Getenv("ADMINSK")
	if ak == "" {
		t.SkipNow()
		return
	}
	assert := assert.New(t)
	c := New(WithAdminAkSk(ak, sk, "http://iovip.qbox.me"))
	ctx, cancel := context.WithTimeout(context.Background(), time.Second*5)
	defer cancel()
	resp, err := c.Get(ctx, Request{URI: "qiniu://1380585377@z0/test/1.png"})
	assert.Nil(err)
	defer resp.Body.Close()
	n, _ := strconv.Atoi(resp.Header.Get("content-length"))
	assert.True(n > 10000)

	resp, err = c.Get(ctx, Request{URI: "qiniu://1380585377@z0/test/qqzzddfff.png"})
	assert.NotNil(err)
	assert.Contains(err.Error(), "Not Found")

	ctx, cancel = context.WithTimeout(context.Background(), time.Microsecond)
	defer cancel()
	resp, err = c.Get(ctx, Request{URI: "qiniu://1380585377@z0/test/1.png"})
	assert.Contains(err.Error(), "context deadline exceeded")

	c = New(WithAdminAkSk(ak, sk, "http://iovip.qbox.me"))
	ctx, cancel = context.WithTimeout(context.Background(), time.Microsecond)
	defer cancel()
	resp, err = c.Get(ctx, Request{URI: "qiniu://1380585377@z0/test/1.png"})
	assert.Contains(err.Error(), "context deadline exceeded")
}

func TestClient_GetFile(t *testing.T) {
	assert := assert.New(t)
	c := New(WithFileHandler())
	ctx := context.Background()
	resp, err := c.Get(ctx, Request{URI: "file://uri.go"})
	assert.Nil(err)
	defer resp.Body.Close()
	assert.True(resp.Size > 100)

	resp, err = c.Get(ctx, Request{URI: "file://./uri.go"})
	assert.Nil(err)
	defer resp.Body.Close()
	assert.True(resp.Size > 100)

	resp, err = c.Get(ctx, Request{URI: "file:///etc/hosts"})
	assert.Nil(err)
	defer resp.Body.Close()
	assert.True(resp.Size > 10)

	_, err = c.Get(ctx, Request{URI: "file://xxxxx.go"})
	assert.Contains(err.Error(), "no such file")
}

func TestClient_GetPublicFile(t *testing.T) {
	assert := assert.New(t)

	s := &http.Server{Addr: ":8877", Handler: http.HandlerFunc(func(rw http.ResponseWriter, req *http.Request) {
		rw.Write([]byte("123"))
	})}
	go s.ListenAndServe()
	time.Sleep(time.Second / 100)
	{
		c := New(WithHTTPHandler())
		ctx, cancel := context.WithTimeout(context.Background(), time.Second*5)
		defer cancel()
		resp, err := c.Get(ctx, Request{URI: "http://localhost:8877/"})
		assert.Nil(err)
		defer resp.Body.Close()
	}

	{
		c := New(WithHTTPPublicHandler())

		ctx := context.Background()
		resp, err := c.Get(ctx, Request{URI: "http://localhost:8877/"})
		assert.NotNil(err)
		assert.Nil(resp)
		assert.Contains(err.Error(), "use of closed network connection")
	}

	s.Close()
}

func TestClient_GetCertHTTPSFile(t *testing.T) {
	assert := assert.New(t)

	ts := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("https 123"))
	}))
	defer ts.Close()

	{
		c := New(WithCertHTTPSHandler([]byte(""), []byte(""), []byte("")))
		ctx, cancel := context.WithTimeout(context.Background(), time.Second*5)
		defer cancel()
		resp, err := c.Get(ctx, Request{URI: ts.URL})
		assert.Nil(err)
		defer resp.Body.Close()
	}
}

func TestClient_GetCertHTTPSFileFromPublic(t *testing.T) {
	if os.Getenv("TRAVIS") != "" {
		t.SkipNow()
	}
	assert := assert.New(t)

	{
		c := New(WithCertHTTPSHandler([]byte(""), []byte(""), []byte("")))
		ctx, cancel := context.WithTimeout(context.Background(), time.Second*5)
		defer cancel()
		resp, err := c.Get(ctx, Request{URI: "https://www.baidu.com/"})
		assert.Nil(err)
		defer resp.Body.Close()
	}

	{
		c := New(WithCertHTTPSHandler([]byte(""), []byte(""), []byte("")))
		ctx, cancel := context.WithTimeout(context.Background(), time.Second*5)
		defer cancel()
		resp, err := c.Get(ctx, Request{URI: "https://www.google.com/"})
		assert.Nil(err)
		defer resp.Body.Close()
	}
}
