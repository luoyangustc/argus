package uri

import (
	"context"
	"github.com/stretchr/testify.v2/assert"
	"os"
	"testing"
	"time"
)

func TestExample(t *testing.T) {
	ak := os.Getenv("AK")
	sk := os.Getenv("SK")
	adminAk := os.Getenv("ADMINAK")
	adminSk := os.Getenv("ADMINSK")
	if ak == "" {
		t.SkipNow()
		return
	}
	assert := assert.New(t)
	ctx, cancel := context.WithTimeout(context.Background(), time.Second*5)
	defer cancel()

	c := New(WithFileHandler(), WithHTTPHandler(), WithUserAkSk(ak, sk, "http://rs.qbox.me"))

	// 读取本地文件
	resp, err := c.Get(ctx, Request{URI: "file://uri.go"})
	assert.Nil(err)
	defer resp.Body.Close()
	assert.True(resp.Size > 100)

	// 读取网络文件
	resp, err = c.Get(ctx, Request{URI: "http://q.hi-hi.cn/1.png"})
	assert.Nil(err)
	defer resp.Body.Close()
	assert.EqualValues(resp.Size, 90273)

	// 通过qiniu协议读取用户自己的文件
	resp, err = c.Get(ctx, Request{URI: "qiniu://1380585377@z0/test/1.png"})
	assert.Nil(err)
	defer resp.Body.Close()
	assert.EqualValues(resp.Size, 90273)

	assert.Len(c.Names(), 4)

	// 通过io读取七牛bucket资源
	// c = New(WithHTTPFromIOHandler("http://iovip.qbox.me"))
	// resp, err = c.Get(ctx, Request{URI: "http://q.hi-hi.cn/1.png"})
	// assert.Nil(err)
	// defer resp.Body.Close()
	// assert.EqualValues(resp.Size, 90273)

	// 通过qiniu协议读取任意用户的文件
	c = New(WithAdminAkSk(adminAk, adminSk, "http://iovip.qbox.me"))
	resp, err = c.Get(ctx, Request{URI: "qiniu://1380585377@z0/test/1.png"})
	assert.Nil(err)
	defer resp.Body.Close()
	assert.EqualValues(resp.Size, 90273)
}
