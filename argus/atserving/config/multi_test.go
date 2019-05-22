package config

import (
	"context"
	"encoding/json"
	"sync"
	"testing"
	"time"

	etcd "github.com/coreos/etcd/clientv3"
	"github.com/stretchr/testify/assert"
)

func TestMultiStaticConfigValue(t *testing.T) {
	var (
		key1 = _ETCD_PREFIX + "/" + "static/6"
		key2 = _ETCD_PREFIX + "/" + "static/7"
	)
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

		var (
			conf1 = NewStaticEtcdValue(kv, key1,
				func(bs []byte) (interface{}, error) {
					var v string
					err := json.Unmarshal(bs, &v)
					return v, err
				},
			)
			conf2 = NewStaticEtcdValue(kv, key2,
				func(bs []byte) (interface{}, error) {
					var v string
					err := json.Unmarshal(bs, &v)
					return v, err
				},
			)
			conf3 = NewMultiStaticConfigValue(conf1, conf2)
		)

		_, err = conf1.Value(context.Background())
		assert.Error(t, ErrConfigNotExist, err)
		_, err = conf2.Value(context.Background())
		assert.Error(t, ErrConfigNotExist, err)
		_, err = conf3.Value(context.Background())
		assert.Error(t, ErrConfigNotExist, err)

		_, err = kv.Put(context.Background(), key2, "\"xx\"")
		assert.NoError(t, err)

		conf1 = NewStaticEtcdValue(kv, key1,
			func(bs []byte) (interface{}, error) {
				var v string
				err := json.Unmarshal(bs, &v)
				return v, err
			},
		)
		conf2 = NewStaticEtcdValue(kv, key2,
			func(bs []byte) (interface{}, error) {
				var v string
				err := json.Unmarshal(bs, &v)
				return v, err
			},
		)
		conf3 = NewMultiStaticConfigValue(conf1, conf2)

		_, err = conf1.Value(context.Background())
		assert.Error(t, ErrConfigNotExist, err)
		value2, err := conf2.Value(context.Background())
		assert.NoError(t, err)
		assert.Equal(t, "xx", value2.(string))
		value3, err := conf3.Value(context.Background())
		assert.NoError(t, err)
		assert.Equal(t, "xx", value3.(string))

		_, err = kv.Put(context.Background(), key1, "\"yy\"")
		assert.NoError(t, err)

		conf1 = NewStaticEtcdValue(kv, key1,
			func(bs []byte) (interface{}, error) {
				var v string
				err := json.Unmarshal(bs, &v)
				return v, err
			},
		)
		conf2 = NewStaticEtcdValue(kv, key2,
			func(bs []byte) (interface{}, error) {
				var v string
				err := json.Unmarshal(bs, &v)
				return v, err
			},
		)
		conf3 = NewMultiStaticConfigValue(conf1, conf2)

		value1, err := conf1.Value(context.Background())
		assert.NoError(t, err)
		assert.Equal(t, "yy", value1.(string))
		value2, err = conf2.Value(context.Background())
		assert.NoError(t, err)
		assert.Equal(t, "xx", value2.(string))
		value3, err = conf3.Value(context.Background())
		assert.NoError(t, err)
		assert.Equal(t, "yy", value3.(string))

	}

}

func TestMultiWatchConfigValue(t *testing.T) {
	var (
		key1 = _ETCD_PREFIX + "/" + "static/8"
		key2 = _ETCD_PREFIX + "/" + "static/9"
	)
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

		var (
			conf1 = NewWatchEtcdValue(context.Background(), cli, key1,
				func(bs []byte) (interface{}, error) {
					var v _MockConfigStruct_2
					err := json.Unmarshal(bs, &v)
					return v, err
				})
			conf2 = NewWatchEtcdValue(context.Background(), cli, key2,
				func(bs []byte) (interface{}, error) {
					var v _MockConfigStruct_2
					err := json.Unmarshal(bs, &v)
					return v, err
				})
			conf3 = NewMultiWatchConfigValue(conf1, conf2)
		)
		_, err = conf1.Value(context.Background())
		assert.Error(t, ErrConfigNotExist, err)
		_, err = conf2.Value(context.Background())
		assert.Error(t, ErrConfigNotExist, err)
		_, err = conf3.Value(context.Background())
		assert.Error(t, ErrConfigNotExist, err)

		var (
			vv1, vv2, vv3 interface{}
			a1, a2, a3    int
			c1, c2, c3    string
			lock          = new(sync.Mutex)
		)

		conf1.Register(
			func(v1 interface{}) error {
				lock.Lock()
				defer lock.Unlock()
				vv1 = v1
				if v1 != nil {
					v2 := v1.(_MockConfigStruct_2)
					a1, c1 = v2.B.A, v2.C
				}
				return nil
			},
		)
		conf2.Register(
			func(v1 interface{}) error {
				lock.Lock()
				defer lock.Unlock()
				vv2 = v1
				if v1 != nil {
					v2 := v1.(_MockConfigStruct_2)
					a2, c2 = v2.B.A, v2.C
				}
				return nil
			},
		)
		conf3.Register(
			func(v1 interface{}) error {
				lock.Lock()
				defer lock.Unlock()
				vv3 = v1
				if v1 != nil {
					v2 := v1.(_MockConfigStruct_2)
					a3, c3 = v2.B.A, v2.C
				}
				return nil
			},
		)

		_, err = kv.Put(context.Background(), key2, "{\"b\": {\"a\": 100}, \"c\": \"xx\"}")
		assert.NoError(t, err)
		time.Sleep(time.Second)

		lock.Lock()
		assert.Nil(t, vv1)
		assert.Equal(t, 100, a2)
		assert.Equal(t, "xx", c2)
		assert.Equal(t, 100, a3)
		assert.Equal(t, "xx", c3)
		lock.Unlock()

		_, err = kv.Delete(context.Background(), key2)
		assert.NoError(t, err)
		time.Sleep(time.Second)

		lock.Lock()
		assert.Nil(t, vv1)
		assert.Nil(t, vv2)
		assert.Nil(t, vv3)
		lock.Unlock()

		_, err = kv.Put(context.Background(), key2, "{\"b\": {\"a\": 100}, \"c\": \"xx\"}")
		assert.NoError(t, err)
		time.Sleep(time.Second)

		lock.Lock()
		assert.Nil(t, vv1)
		assert.Equal(t, 100, a2)
		assert.Equal(t, "xx", c2)
		assert.Equal(t, 100, a3)
		assert.Equal(t, "xx", c3)
		lock.Unlock()

		_, err = kv.Put(context.Background(), key1, "{\"b\": {\"a\": 101}, \"c\": \"yy\"}")
		assert.NoError(t, err)
		time.Sleep(time.Second)

		lock.Lock()
		assert.Equal(t, 101, a1)
		assert.Equal(t, "yy", c1)
		assert.Equal(t, 100, a2)
		assert.Equal(t, "xx", c2)
		assert.Equal(t, 101, a3)
		assert.Equal(t, "yy", c3)
		lock.Unlock()

		_, err = kv.Delete(context.Background(), key2)
		assert.NoError(t, err)
		time.Sleep(time.Second)

		lock.Lock()
		assert.Equal(t, 101, a1)
		assert.Equal(t, "yy", c1)
		assert.Nil(t, vv2)
		assert.Equal(t, 101, a3)
		assert.Equal(t, "yy", c3)
		lock.Unlock()

	}

}
