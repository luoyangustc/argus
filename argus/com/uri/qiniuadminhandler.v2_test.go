package uri

import (
	"bytes"
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify.v2/assert"
)

func TestFoo(t *testing.T) {

	// resp, err := WithAdminAkSkV2(
	// 	QiniuAdminHandlerConfig{
	// 		RSHost: "rs.qbox.me", IOHosts: map[string]string{"z0": "iovip.qbox.me"},
	// 	}).
	// 	Get(context.Background(), Request{URI: "qiniu://1380702424@z0/ava-test-1/lena_std.jpg"})
	// assert.NoError(t, err)
	// defer resp.Body.Close()

	// bs, _ := ioutil.ReadAll(resp.Body)
	// t.Fatalf("%d\n", len(bs))

}

//----------------------------------------------------------------------------//

func TestCache(t *testing.T) {

	var v interface{}
	var err error
	var fetch = func(context.Context, _Key) (interface{}, error) { return v, err }

	var c = newCache(CacheConfig{
		Duration:    1,
		ChanBufSize: 1,
	}, fetch)

	{
		v = struct{ A int }{A: 100}
		v2, err := c.Get(context.Background(), bytes.NewBufferString("AA"))
		assert.NoError(t, err)
		assert.Equal(t, 100, v2.(struct{ A int }).A)
	}
	time.Sleep(time.Millisecond * 2)
	{
		v = struct{ A int }{A: 99}
		v2, err := c.Get(context.Background(), bytes.NewBufferString("BB"))
		assert.NoError(t, err)
		assert.Equal(t, 99, v2.(struct{ A int }).A)
	}
	time.Sleep(time.Millisecond * 2)
	{
		v2, err := c.Get(context.Background(), bytes.NewBufferString("AA"))
		assert.NoError(t, err)
		assert.Equal(t, 100, v2.(struct{ A int }).A)
	}

}

func BenchmarkCache(b *testing.B) {
	var v interface{} = struct{ A int }{A: 100}
	var err error
	var fetch = func(context.Context, _Key) (interface{}, error) {
		time.Sleep(time.Millisecond)
		return v, err
	}

	var c = newCache(CacheConfig{Duration: 100}, fetch)

	for i := 0; i < b.N; i++ {
		c.Get(context.Background(), bytes.NewBufferString("AA"))
	}
}
