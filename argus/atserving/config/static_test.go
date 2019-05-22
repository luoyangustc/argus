package config

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	etcd "github.com/coreos/etcd/clientv3"
	"github.com/stretchr/testify/assert"
)

func TestStaticEtcd(t *testing.T) {
	var key = _ETCD_PREFIX + "/" + "static/1"
	cleanETCD(t, key)

	{
		cli, err := etcd.New(
			etcd.Config{
				Endpoints:   _ETCD_HOST,
				DialTimeout: time.Second,
			},
		)
		assert.NoError(t, err)
		kv := etcd.NewKV(cli)

		resp, err := kv.Get(context.Background(), key)
		assert.NoError(t, err)
		assert.Empty(t, resp.Kvs)

		_, err = kv.Put(context.Background(), key, "xx")
		assert.NoError(t, err)

		resp, err = kv.Get(context.Background(), key)
		assert.NoError(t, err)
		assert.Equal(t, 1, len(resp.Kvs))
		assert.Equal(t, "xx", string(resp.Kvs[0].Value))

		_, err = kv.Put(context.Background(), key, "yy")
		assert.NoError(t, err)

		resp, err = kv.Get(context.Background(), key)
		assert.NoError(t, err)
		assert.Equal(t, 1, len(resp.Kvs))
		assert.Equal(t, "yy", string(resp.Kvs[0].Value))
	}
}

func TestStaticEtcdValue(t *testing.T) {
	var key = _ETCD_PREFIX + "/" + "static/2"
	cleanETCD(t, key)

	{
		cli, err := etcd.New(
			etcd.Config{
				Endpoints:   _ETCD_HOST,
				DialTimeout: time.Second,
			},
		)
		assert.NoError(t, err)
		kv := etcd.NewKV(cli)

		conf := NewStaticEtcdValue(kv, key,
			func(bs []byte) (interface{}, error) {
				var v string
				err := json.Unmarshal(bs, &v)
				return v, err
			},
		)
		_, err = conf.Value(context.Background())
		assert.Error(t, ErrConfigNotExist, err)

		_, err = kv.Put(context.Background(), key, "\"xx\"")
		assert.NoError(t, err)

		_, err = conf.Value(context.Background())
		assert.Error(t, ErrConfigNotExist, err)

		conf = NewStaticEtcdValue(kv, key,
			func(bs []byte) (interface{}, error) {
				var v string
				err := json.Unmarshal(bs, &v)
				return v, err
			},
		)
		value1, err := conf.Value(context.Background())
		assert.NoError(t, err)
		assert.Equal(t, "xx", value1.(string))

		_, err = kv.Put(context.Background(), key, "\"yy\"")
		assert.NoError(t, err)

		value2, err := conf.Value(context.Background())
		assert.NoError(t, err)
		assert.Equal(t, "xx", value2.(string))

		conf = NewStaticEtcdValue(kv, key,
			func(bs []byte) (interface{}, error) {
				var v string
				err := json.Unmarshal(bs, &v)
				return v, err
			},
		)
		value3, err := conf.Value(context.Background())
		assert.NoError(t, err)
		assert.Equal(t, "yy", value3.(string))

	}
}

type _MockConfigStruct_1 struct {
	A int `json:"a"`
}

type _MockConfigStruct_2 struct {
	B _MockConfigStruct_1 `json:"b"`
	C string              `json:"c"`
}

func TestStaticEtcdChildValue(t *testing.T) {
	var key = _ETCD_PREFIX + "/" + "static/3"
	cleanETCD(t, key)

	{
		cli, err := etcd.New(
			etcd.Config{
				Endpoints:   _ETCD_HOST,
				DialTimeout: time.Second,
			},
		)
		assert.NoError(t, err)
		kv := etcd.NewKV(cli)

		conf1 := NewStaticEtcdValue(kv, key,
			func(bs []byte) (interface{}, error) {
				var v _MockConfigStruct_2
				err := json.Unmarshal(bs, &v)
				return v, err
			},
		)
		_, err = conf1.Value(context.Background())
		assert.Error(t, ErrConfigNotExist, err)
		conf2 := conf1.Find(func(v interface{}) interface{} { return v.(_MockConfigStruct_2).B })
		_, err = conf2.Value(context.Background())
		assert.Error(t, ErrConfigNotExist, err)
		conf3 := conf2.Find(func(v interface{}) interface{} { return v.(_MockConfigStruct_1).A })
		_, err = conf3.Value(context.Background())
		assert.Error(t, ErrConfigNotExist, err)

		_, err = kv.Put(context.Background(), key, "{\"b\": {\"a\": 100}, \"c\": \"xx\"}")
		assert.NoError(t, err)

		conf1 = NewStaticEtcdValue(kv, key,
			func(bs []byte) (interface{}, error) {
				var v _MockConfigStruct_2
				err := json.Unmarshal(bs, &v)
				return v, err
			},
		)

		v1, err := conf1.Value(context.Background())
		assert.NoError(t, err)
		assert.Equal(t, "xx", v1.(_MockConfigStruct_2).C)
		conf2 = conf1.Find(func(v interface{}) interface{} { return v.(_MockConfigStruct_2).B })
		v2, err := conf2.Value(context.Background())
		assert.NoError(t, err)
		assert.Equal(t, 100, v2.(_MockConfigStruct_1).A)
		conf3 = conf2.Find(func(v interface{}) interface{} { return v.(_MockConfigStruct_1).A })
		v3, err := conf3.Value(context.Background())
		assert.NoError(t, err)
		assert.Equal(t, 100, v3.(int))

		_, err = kv.Put(context.Background(), key, "{\"b\": {\"a\": 101}, \"c\": \"yy\"}")
		assert.NoError(t, err)

		v1, err = conf1.Value(context.Background())
		assert.NoError(t, err)
		assert.Equal(t, "xx", v1.(_MockConfigStruct_2).C)
		v2, err = conf2.Value(context.Background())
		assert.NoError(t, err)
		assert.Equal(t, 100, v2.(_MockConfigStruct_1).A)
		v3, err = conf3.Value(context.Background())
		assert.NoError(t, err)
		assert.Equal(t, 100, v3.(int))

		conf1 = NewStaticEtcdValue(kv, key,
			func(bs []byte) (interface{}, error) {
				var v _MockConfigStruct_2
				err := json.Unmarshal(bs, &v)
				return v, err
			},
		)

		v1, err = conf1.Value(context.Background())
		assert.NoError(t, err)
		assert.Equal(t, "yy", v1.(_MockConfigStruct_2).C)
		conf2 = conf1.Find(func(v interface{}) interface{} { return v.(_MockConfigStruct_2).B })
		v2, err = conf2.Value(context.Background())
		assert.NoError(t, err)
		assert.Equal(t, 101, v2.(_MockConfigStruct_1).A)
		conf3 = conf2.Find(func(v interface{}) interface{} { return v.(_MockConfigStruct_1).A })
		v3, err = conf3.Value(context.Background())
		assert.NoError(t, err)
		assert.Equal(t, 101, v3.(int))

	}
}
