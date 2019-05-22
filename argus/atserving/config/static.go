package config

import (
	"context"
	"sync"

	etcd "github.com/coreos/etcd/clientv3"
	xlog "github.com/qiniu/xlog.v1"
)

//----------------------------------------------------------------------------//

type staticConfigValue struct {
	v interface{}
	f func() interface{}
}

func NewStaticConfigValue(f func() interface{}) StaticConfigValue { return &staticConfigValue{f: f} }
func (v *staticConfigValue) Value(ctx context.Context) (interface{}, error) {
	if v.v != nil {
		return v.v, nil
	}
	v.v = v.f()
	if v.v == nil {
		return nil, ErrConfigNotExist
	}
	return v.v, nil
}
func (v *staticConfigValue) Find(find func(interface{}) interface{}) StaticConfigValue {
	return NewStaticConfigValue(
		func() interface{} {
			vv, _ := v.Value(context.Background())
			if vv == nil {
				return nil
			}
			return find(vv)
		},
	)
}

//----------------------------------------------------------------------------//

type staticEtcdValue struct {
	etcd.KV

	key   string
	value interface{}
	err   error

	parseValue func([]byte) (interface{}, error)
	*sync.Mutex
}

func NewStaticEtcdValue(
	client etcd.KV, key string,
	parseValue func([]byte) (interface{}, error),
) StaticConfigValue {
	return &staticEtcdValue{
		KV:         client,
		key:        key,
		parseValue: parseValue,
		Mutex:      new(sync.Mutex),
	}
}

func (v *staticEtcdValue) fetch(ctx context.Context) error {
	v.Lock()
	defer v.Unlock()

	xl := xlog.FromContextSafe(ctx)

	if v.value != nil || v.err != nil {
		return v.err
	}

	var resp *etcd.GetResponse
	resp, v.err = v.KV.Get(ctx, v.key)
	if v.err != nil {
		xl.Warnf("get etcd failed. %s %v", v.key, v.err)
		return v.err
	}
	if resp.Kvs == nil || len(resp.Kvs) == 0 {
		xl.Warnf("get etcd failed. %s %#v", v.key, resp)
		v.err = ErrConfigNotExist
		return v.err
	}
	v.value, v.err = v.parseValue(resp.Kvs[0].Value)
	return v.err
}

func (v *staticEtcdValue) Value(ctx context.Context) (interface{}, error) {
	if err := v.fetch(ctx); err != nil {
		return nil, err
	}
	return v.value, nil
}

func (v *staticEtcdValue) Find(find func(interface{}) interface{}) StaticConfigValue {
	return staticChildConfigValue{
		ConfigValue: v,
		find:        find,
	}
}

type staticChildConfigValue struct {
	ConfigValue
	find func(interface{}) interface{}
}

func (v staticChildConfigValue) Value(ctx context.Context) (interface{}, error) {
	value, err := v.ConfigValue.Value(ctx)
	if err != nil {
		return nil, err
	}
	return v.find(value), nil
}

func (v staticChildConfigValue) Find(find func(interface{}) interface{}) StaticConfigValue {
	return staticChildConfigValue{
		ConfigValue: v,
		find:        find,
	}
}
