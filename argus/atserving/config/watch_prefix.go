package config

import (
	"context"
	"sync"

	etcd "github.com/coreos/etcd/clientv3"
	"github.com/coreos/etcd/mvcc/mvccpb"

	"github.com/qiniu/xlog.v1"
)

type watchPrefixEtcdValues struct {
	etcd.KV
	etcd.Watcher
	wChan etcd.WatchChan

	prefix string
	parse  func(context.Context, []byte, []byte) (interface{}, interface{}, error)
	sets   []func(interface{}, interface{}) error

	*sync.Mutex
}

func NewWatchPrefixEtcdValues(
	ctx context.Context,
	client *etcd.Client,
	prefix string,
	parseKV func(context.Context, []byte, []byte) (interface{}, interface{}, error),
) WatchPrefixConfigValues {

	values := &watchPrefixEtcdValues{
		KV:      etcd.NewKV(client),
		Watcher: etcd.NewWatcher(client),
		prefix:  prefix,
		parse:   parseKV,
		sets:    make([]func(interface{}, interface{}) error, 0),
		Mutex:   new(sync.Mutex),
	}
	values.wChan = values.Watch(ctx, prefix, etcd.WithPrefix())
	go values.runWatch(ctx)

	return values
}

func (v *watchPrefixEtcdValues) runWatch(ctx context.Context) {

	for resp := range v.wChan {
		var (
			xl  = xlog.NewDummy()
			ctx = xlog.NewContext(ctx, xl)
		)
		if len(resp.Events) == 0 {
			continue
		}
		for _, event := range resp.Events {
			var (
				key, value interface{}
				err        error
			)
			if event.Type == mvccpb.DELETE {
				key, _, err = v.parse(ctx, event.Kv.Key, nil)
			} else {
				key, value, err = v.parse(ctx, event.Kv.Key, event.Kv.Value)
			}
			if err != nil {
				continue
			}

			func() {
				v.Lock()
				defer v.Unlock()

				for _, set := range v.sets {
					if err = set(key, value); err != nil {
						xl.Warnf("set config value failed. %v", err)
					}
				}
			}()

		}
	}

}

func (v *watchPrefixEtcdValues) Values(ctx context.Context) (
	keys []interface{}, values []interface{}, err error) {

	keys = make([]interface{}, 0)
	values = make([]interface{}, 0)

	var (
		xl   = xlog.FromContextSafe(ctx)
		resp *etcd.GetResponse
	)
	resp, err = v.KV.Get(ctx, v.prefix, etcd.WithPrefix())
	if err != nil {
		xl.Warnf("get etcd failed. %s %v", v.prefix, err)
		return
	}
	if resp.Kvs == nil || len(resp.Kvs) == 0 {
		return
	}
	for _, kv := range resp.Kvs {
		var key, value interface{}
		if key, value, err = v.parse(ctx, kv.Key, kv.Value); err != nil {
			return
		}
		keys = append(keys, key)
		values = append(values, value)
	}
	return
}

func (v *watchPrefixEtcdValues) Register(set func(interface{}, interface{}) error) {
	v.Lock()
	defer v.Unlock()
	v.sets = append(v.sets, set)
}
