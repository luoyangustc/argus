package config

import (
	"context"
	"encoding/json"
	"sync"
	"testing"
	"time"

	"strings"

	etcd "github.com/coreos/etcd/clientv3"
	"github.com/stretchr/testify/assert"
)

type _MockConfigKey_1 struct {
	K1 string
	K2 string
}

func (key *_MockConfigKey_1) Parse(bs []byte) error {
	var s = string(bs)
	var strs = strings.Split(s, "/")
	key.K1 = strs[len(strs)-2]
	key.K2 = strs[len(strs)-1]
	return nil
}

func TestWatchPrefixEtcdValue(t *testing.T) {
	var key = _ETCD_PREFIX + "/" + "static/10"
	var key1 = key + "/1"
	var key2 = key + "/2"
	cleanETCD(t, key1, key2)

	{
		cli, err := etcd.New(
			etcd.Config{
				Endpoints:   _ETCD_HOST,
				DialTimeout: time.Second,
			},
		)
		assert.NoError(t, err)
		kv := etcd.NewKV(cli)

		conf1 := NewWatchPrefixEtcdValues(context.Background(), cli, key,
			func(ctx context.Context, kbs, vbs []byte) (interface{}, interface{}, error) {
				var k _MockConfigKey_1
				var v _MockConfigStruct_2
				if err := (&k).Parse(kbs); err != nil {
					return nil, nil, err
				}
				if vbs == nil {
					return k, nil, nil
				}
				err := json.Unmarshal(vbs, &v)
				return k, v, err
			})
		_, _, err = conf1.Values(context.Background())
		assert.NoError(t, err)

		_, err = kv.Put(context.Background(), key1, "{\"b\": {\"a\": 100}, \"c\": \"xx\"}")
		assert.NoError(t, err)
		time.Sleep(time.Second)

		ks, vs, err := conf1.Values(context.Background())
		assert.NoError(t, err)
		assert.Equal(t, "10", ks[0].(_MockConfigKey_1).K1)
		assert.Equal(t, "1", ks[0].(_MockConfigKey_1).K2)
		assert.Equal(t, "xx", vs[0].(_MockConfigStruct_2).C)

		_, err = kv.Put(context.Background(), key1, "{\"b\": {\"a\": 101}, \"c\": \"yy\"}")
		assert.NoError(t, err)
		_, err = kv.Put(context.Background(), key2, "{\"b\": {\"a\": 102}, \"c\": \"zz\"}")
		assert.NoError(t, err)
		time.Sleep(time.Second)

		ks, vs, err = conf1.Values(context.Background())
		assert.NoError(t, err)
		assert.Equal(t, "10", ks[0].(_MockConfigKey_1).K1)
		assert.Equal(t, "1", ks[0].(_MockConfigKey_1).K2)
		assert.Equal(t, "yy", vs[0].(_MockConfigStruct_2).C)
		assert.Equal(t, "10", ks[1].(_MockConfigKey_1).K1)
		assert.Equal(t, "2", ks[1].(_MockConfigKey_1).K2)
		assert.Equal(t, "zz", vs[1].(_MockConfigStruct_2).C)

		_, err = kv.Delete(context.Background(), key1)
		assert.NoError(t, err)
		_, err = kv.Delete(context.Background(), key2)
		assert.NoError(t, err)
		time.Sleep(time.Second)

		ks, vs, err = conf1.Values(context.Background())
		assert.NoError(t, err)
		assert.Empty(t, ks)
		assert.Empty(t, vs)

		var (
			k1, k2 = _MockConfigKey_1{K1: "10", K2: "1"}, _MockConfigKey_1{K1: "10", K2: "2"}
			v1, v2 bool
			a1, a2 int
			c1, c2 string
			lock   = new(sync.Mutex)
		)
		conf1.Register(
			func(kk1, vv1 interface{}) error {
				lock.Lock()
				defer lock.Unlock()
				kk2 := kk1.(_MockConfigKey_1)
				switch kk2.K2 {
				case k1.K2:
					if vv1 == nil {
						v1 = false
						return nil
					}
					v1 = true
					vv2 := vv1.(_MockConfigStruct_2)
					a1, c1 = vv2.B.A, vv2.C
				case k2.K2:
					if vv1 == nil {
						v2 = false
						return nil
					}
					v2 = true
					vv2 := vv1.(_MockConfigStruct_2)
					a2, c2 = vv2.B.A, vv2.C
				}
				return nil
			},
		)

		_, err = kv.Put(context.Background(), key1, "{\"b\": {\"a\": 100}, \"c\": \"xx\"}")
		assert.NoError(t, err)
		_, err = kv.Put(context.Background(), key2, "{\"b\": {\"a\": 101}, \"c\": \"yy\"}")
		assert.NoError(t, err)
		time.Sleep(time.Second)

		lock.Lock()
		assert.True(t, v1)
		assert.Equal(t, 100, a1)
		assert.Equal(t, "xx", c1)
		assert.True(t, v2)
		assert.Equal(t, 101, a2)
		assert.Equal(t, "yy", c2)
		lock.Unlock()

		_, err = kv.Delete(context.Background(), key1)
		assert.NoError(t, err)
		time.Sleep(time.Second)

		lock.Lock()
		assert.False(t, v1)
		assert.True(t, v2)
		assert.Equal(t, 101, a2)
		assert.Equal(t, "yy", c2)
		lock.Unlock()

		_, err = kv.Put(context.Background(), key1, "{\"b\": {\"a\": 102}, \"c\": \"xx\"}")
		assert.NoError(t, err)
		time.Sleep(time.Second)

		lock.Lock()
		assert.True(t, v1)
		assert.Equal(t, 102, a1)
		assert.Equal(t, "xx", c1)
		assert.True(t, v2)
		assert.Equal(t, 101, a2)
		assert.Equal(t, "yy", c2)
		lock.Unlock()

		_, err = kv.Put(context.Background(), key1, "{\"b\": {\"a\": 103}, \"c\": \"xx\"}")
		assert.NoError(t, err)
		_, err = kv.Put(context.Background(), key2, "{\"b\": {\"a\": 104}, \"c\": \"yy\"}")
		assert.NoError(t, err)
		time.Sleep(time.Second)

		lock.Lock()
		assert.True(t, v1)
		assert.Equal(t, 103, a1)
		assert.Equal(t, "xx", c1)
		assert.True(t, v2)
		assert.Equal(t, 104, a2)
		assert.Equal(t, "yy", c2)
		lock.Unlock()

	}
}
