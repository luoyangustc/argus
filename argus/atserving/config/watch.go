package config

import (
	"context"
	"reflect"
	"sync"

	etcd "github.com/coreos/etcd/clientv3"
	"github.com/coreos/etcd/mvcc/mvccpb"

	"github.com/qiniu/xlog.v1"
)

type watchEtcdValue struct {
	*staticEtcdValue
	etcd.Watcher

	wChan etcd.WatchChan
	sets  []func(interface{}) error
}

func NewWatchEtcdValue(
	ctx context.Context,
	client *etcd.Client,
	key string,
	parseValue func([]byte) (interface{}, error),
) WatchConfigValue {

	value := &watchEtcdValue{
		staticEtcdValue: &staticEtcdValue{
			KV:         etcd.NewKV(client),
			key:        key,
			parseValue: parseValue,
			Mutex:      new(sync.Mutex),
		},
		Watcher: etcd.NewWatcher(client),
		sets:    make([]func(interface{}) error, 0),
	}
	value.wChan = value.Watch(ctx, key)
	go value.runWatch(ctx)

	return value
}

func (v *watchEtcdValue) runWatch(ctx context.Context) {

	xl := xlog.FromContextSafe(ctx)

	for resp := range v.wChan {
		if len(resp.Events) == 0 {
			xl.Warnf("etcd value not found. %s", v.key)
			continue
		}
		var (
			event = resp.Events[len(resp.Events)-1]
			value interface{}
			err   error
		)

		if event.Type == mvccpb.DELETE {
			// NOTHTING TO DO
		} else {
			if value, err = v.parseValue(event.Kv.Value); err != nil {
				xl.Warnf("etcd value unmarshal failed. %s %s", v.key, string(event.Kv.Value))
				v = nil
			}
		}
		func() {
			v.Lock()
			defer v.Unlock()

			if v.err = err; v.err != nil {
				return
			}
			if !reflect.DeepEqual(v.value, value) {
				v.value = value
				for _, set := range v.sets {
					if err := set(v.value); err != nil {
						xl.Warnf("set config value failed. %v", err)
					}
				}
			}
		}()
	}

}

func (v *watchEtcdValue) Value(ctx context.Context) (interface{}, error) {
	return v.staticEtcdValue.Value(ctx)
}

func (v *watchEtcdValue) Register(set func(interface{}) error) {
	v.Lock()
	defer v.Unlock()
	v.sets = append(v.sets, set)
}

func (v *watchEtcdValue) Find(find func(interface{}) interface{}) WatchConfigValue {
	return &watchChildConfigValue{
		WatchConfigValue: v,
		find:             find,
		values:           make([]interface{}, 0),
	}
}

type watchChildConfigValue struct {
	WatchConfigValue
	find   func(interface{}) interface{}
	values []interface{}
}

func (v *watchChildConfigValue) Value(ctx context.Context) (interface{}, error) {
	value, err := v.WatchConfigValue.Value(ctx)
	if err != nil {
		return nil, err
	}
	return v.find(value), nil
}

func (v *watchChildConfigValue) Register(set func(interface{}) error) {
	index := len(v.values)
	v.values = append(v.values, nil)
	v.WatchConfigValue.Register(
		func(v1 interface{}) error {
			v2 := v.find(v1)
			if v.values[index] == nil || !reflect.DeepEqual(v.values[index], v2) {
				v.values[index] = v2
				return set(v2)
			}
			return nil
		},
	)
}

func (v *watchChildConfigValue) Find(find func(interface{}) interface{}) WatchConfigValue {
	return &watchChildConfigValue{
		WatchConfigValue: v,
		find:             find,
		values:           make([]interface{}, 0),
	}
}
