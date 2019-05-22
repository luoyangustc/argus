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

var _ WatchConfigValue = &_MockWatchConfigValue{}

type _MockWatchConfigValue struct {
	value interface{}
	err   error
	sets  []func(interface{}) error
}

func newMockWatchConfigValue() *_MockWatchConfigValue {
	return &_MockWatchConfigValue{sets: make([]func(interface{}) error, 0)}
}
func (v *_MockWatchConfigValue) Value(ctx context.Context) (interface{}, error) { return v.value, v.err }
func (v *_MockWatchConfigValue) Register(set func(interface{}) error)           { v.sets = append(v.sets, set) }
func (v *_MockWatchConfigValue) Find(find func(interface{}) interface{}) WatchConfigValue {
	return &watchChildConfigValue{
		WatchConfigValue: v,
		find:             find,
		values:           make([]interface{}, 0),
	}
}

func TestWatchEtcdValue(t *testing.T) {
	var key1 = _ETCD_PREFIX + "/" + "static/4"
	var key2 = _ETCD_PREFIX + "/" + "static/5"
	cleanETCD(t, key1, key2)

	{
		var key = key1
		cli, err := etcd.New(
			etcd.Config{
				Endpoints:   _ETCD_HOST,
				DialTimeout: time.Second,
			},
		)
		assert.NoError(t, err)
		kv := etcd.NewKV(cli)

		conf1 := NewWatchEtcdValue(context.Background(), cli, key,
			func(bs []byte) (interface{}, error) {
				var v _MockConfigStruct_2
				err := json.Unmarshal(bs, &v)
				return v, err
			})
		_, err = conf1.Value(context.Background())
		assert.Error(t, ErrConfigNotExist, err)

		_, err = kv.Put(context.Background(), key, "{\"b\": {\"a\": 100}, \"c\": \"xx\"}")
		assert.NoError(t, err)
		time.Sleep(time.Second)

		v1, err := conf1.Value(context.Background())
		assert.NoError(t, err)
		assert.Equal(t, "xx", v1.(_MockConfigStruct_2).C)

		var (
			a1   int
			c1   string
			lock = new(sync.Mutex)
		)
		conf1.Register(
			func(v1 interface{}) error {
				lock.Lock()
				defer lock.Unlock()
				v2 := v1.(_MockConfigStruct_2)
				a1, c1 = v2.B.A, v2.C
				return nil
			},
		)

		_, err = kv.Put(context.Background(), key, "{\"b\": {\"a\": 101}, \"c\": \"yy\"}")
		assert.NoError(t, err)
		time.Sleep(time.Second)

		v1, err = conf1.Value(context.Background())
		assert.NoError(t, err)
		assert.Equal(t, "yy", v1.(_MockConfigStruct_2).C)
		lock.Lock()
		assert.Equal(t, 101, a1)
		assert.Equal(t, "yy", c1)
		lock.Unlock()
	}
	{
		var key = key2
		cli, err := etcd.New(
			etcd.Config{
				Endpoints:   _ETCD_HOST,
				DialTimeout: time.Second,
			},
		)
		assert.NoError(t, err)
		kv := etcd.NewKV(cli)

		var (
			conf1 = NewWatchEtcdValue(context.Background(), cli, key,
				func(bs []byte) (interface{}, error) {
					var v _MockConfigStruct_2
					err := json.Unmarshal(bs, &v)
					return v, err
				})
			conf2 = conf1.Find(func(v interface{}) interface{} { return v.(_MockConfigStruct_2).B })
			conf3 = conf2.Find(func(v interface{}) interface{} { return v.(_MockConfigStruct_1).A })

			a1 int
			c1 string
			a2 int
			a3 int

			lock = new(sync.Mutex)
		)

		conf1.Register(
			func(v1 interface{}) error {
				lock.Lock()
				defer lock.Unlock()
				v2 := v1.(_MockConfigStruct_2)
				a1, c1 = v2.B.A, v2.C
				return nil
			},
		)
		conf2.Register(
			func(v1 interface{}) error {
				lock.Lock()
				defer lock.Unlock()
				v2 := v1.(_MockConfigStruct_1)
				a2 = v2.A
				return nil
			},
		)
		conf3.Register(
			func(v1 interface{}) error {
				lock.Lock()
				defer lock.Unlock()
				v2 := v1.(int)
				a3 = v2
				return nil
			},
		)

		_, err = conf1.Value(context.Background())
		assert.Error(t, ErrConfigNotExist, err)
		_, err = conf2.Value(context.Background())
		assert.Error(t, ErrConfigNotExist, err)
		_, err = conf3.Value(context.Background())
		assert.Error(t, ErrConfigNotExist, err)

		_, err = kv.Put(context.Background(), key, "{\"b\": {\"a\": 100}, \"c\": \"xx\"}")
		assert.NoError(t, err)
		time.Sleep(time.Second)

		v1, err := conf1.Value(context.Background())
		assert.NoError(t, err)
		assert.Equal(t, "xx", v1.(_MockConfigStruct_2).C)
		v2, err := conf2.Value(context.Background())
		assert.NoError(t, err)
		assert.Equal(t, 100, v2.(_MockConfigStruct_1).A)
		v3, err := conf3.Value(context.Background())
		assert.NoError(t, err)
		assert.Equal(t, 100, v3.(int))

		lock.Lock()
		assert.Equal(t, 100, a1)
		assert.Equal(t, "xx", c1)
		assert.Equal(t, 100, a2)
		assert.Equal(t, 100, a3)
		lock.Unlock()

		_, err = kv.Put(context.Background(), key, "{\"b\": {\"a\": 101}, \"c\": \"yy\"}")
		assert.NoError(t, err)
		time.Sleep(time.Second)

		v1, err = conf1.Value(context.Background())
		assert.NoError(t, err)
		assert.Equal(t, "yy", v1.(_MockConfigStruct_2).C)
		v2, err = conf2.Value(context.Background())
		assert.NoError(t, err)
		assert.Equal(t, 101, v2.(_MockConfigStruct_1).A)
		v3, err = conf3.Value(context.Background())
		assert.NoError(t, err)
		assert.Equal(t, 101, v3.(int))

		lock.Lock()
		assert.Equal(t, 101, a1)
		assert.Equal(t, "yy", c1)
		assert.Equal(t, 101, a2)
		assert.Equal(t, 101, a3)
		lock.Unlock()
	}

}

func TestWatchChildValue(t *testing.T) {

	{
		var (
			conf1 = newMockWatchConfigValue()
			conf2 = conf1.Find(func(v interface{}) interface{} { return v.(_MockConfigStruct_2).B })
			conf3 = conf2.Find(func(v interface{}) interface{} { return v.(_MockConfigStruct_1).A })

			a1 int
			c1 string
			a2 int
			a3 int
		)

		conf1.Register(
			func(v1 interface{}) error {
				v2 := v1.(_MockConfigStruct_2)
				a1, c1 = v2.B.A, v2.C
				return nil
			},
		)
		conf2.Register(
			func(v1 interface{}) error {
				v2 := v1.(_MockConfigStruct_1)
				a2 = v2.A
				return nil
			},
		)
		conf3.Register(
			func(v1 interface{}) error {
				v2 := v1.(int)
				a3 = v2
				return nil
			},
		)

		for _, set := range conf1.sets {
			set(
				_MockConfigStruct_2{
					B: _MockConfigStruct_1{
						A: 100,
					},
					C: "xx",
				},
			)
		}

		assert.Equal(t, 100, a1)
		assert.Equal(t, "xx", c1)
		assert.Equal(t, 100, a2)
		assert.Equal(t, 100, a3)
	}
}
