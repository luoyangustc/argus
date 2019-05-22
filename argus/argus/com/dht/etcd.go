package dht

import (
	"bytes"
	"context"
	"fmt"
	"strconv"
	"sync"
	"time"

	etcd "github.com/coreos/etcd/clientv3"
	"github.com/coreos/etcd/clientv3/concurrency"

	"github.com/qiniu/xlog.v1"
	"qbox.us/dht"
)

type ETCD struct {
	*etcd.Client
	Prefix string

	sync.Locker
}

func NewETCD(cli *etcd.Client, prefix string) ETCD {
	session, _ := concurrency.NewSession(cli)
	e := ETCD{Client: cli, Prefix: prefix}
	e.Locker = concurrency.NewLocker(session, e.lockKey())
	return e
}

func (e ETCD) lockKey() string          { return e.Prefix + "/" + "LOCK" }
func (e ETCD) nodePrefix() string       { return e.Prefix + "/node/" }
func (e ETCD) nodeKey(index int) string { return e.nodePrefix() + strconv.Itoa(index) }

func (e ETCD) Register(ctx context.Context, host string, ttl int64) error {
	xl := xlog.FromContextSafe(ctx)
	_ = xl

	e.Lock()
	defer e.Unlock()

	var key string
	var i = 1
	for {
		key = e.nodeKey(i)
		resp, err := etcd.NewKV(e.Client).Get(ctx, key)
		if err != nil {
			xl.Warnf("get etcd failed. %s %v", key, err)
			return err
		}
		if resp.Kvs == nil || len(resp.Kvs) == 0 {
			break
		}
		if string(resp.Kvs[0].Value) == host {
			break
		}
		i++
	}

	var lease = etcd.NewLease(e.Client)

	leaseResp, err := lease.Grant(ctx, ttl)
	if err != nil {
		panic(err)
	}
	var id = leaseResp.ID

	_, err = etcd.NewKV(e.Client).Put(ctx, key, host, etcd.WithLease(id))
	if err != nil {
		panic(err)
	}

	ctx, cancel := context.WithCancel(ctx)
	go func() {
		defer cancel()
		ch, err := lease.KeepAlive(ctx, id)
		if err != nil {
			_ = err
		}
		for {
			select {
			case <-ctx.Done():
				return
			case _, ok := <-ch:
				if ok {
				} else {
					if ctx.Err() == nil {
						panic("keepalive failed")
					}
					return
				}
			}
		}
	}()
	go func() { // HeartBeat
		ticker := time.NewTicker(time.Second * time.Duration(ttl-1))
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
			}

			resp, _ := etcd.NewKV(e.Client).Get(ctx, key)
			if resp.Kvs == nil || len(resp.Kvs) == 0 {
				panic(fmt.Sprintf("%s != nil", host))
			}
			if string(resp.Kvs[0].Value) != host {
				panic(fmt.Sprintf("%s != %s", string(resp.Kvs[0].Value), host))
			}

			// _, err := lease.KeepAliveOnce(ctx, id)
			// if err != nil {
			// 	panic(err)
			// }
		}
	}()
	return nil
}

func (e ETCD) Nodes(ctx context.Context, hook func(dht.NodeInfos) error) (dht.NodeInfos, error) {

	var (
		xl     = xlog.FromContextSafe(ctx)
		prefix = e.nodePrefix()
		nodes  = []dht.NodeInfo{}
		locker = new(sync.Mutex)
	)

	if hook != nil {
		watcher := etcd.NewWatcher(e.Client)
		ch := watcher.Watch(ctx, prefix, etcd.WithPrefix())
		go func() {

			put := func(key []byte, host string) {
				if host == "" {
					return
				}
				locker.Lock()
				defer locker.Unlock()

				for i, node := range nodes {
					if bytes.Equal(node.Key, key) {
						nodes[i].Host = host
						_ = hook(nodes)
						return
					}
				}

				nodes = append(nodes, dht.NodeInfo{Key: key, Host: host})
				_ = hook(nodes)
			}
			del := func(key []byte) {
				locker.Lock()
				defer locker.Unlock()

				for i, node := range nodes {
					if bytes.Equal(node.Key, key) {
						nodes[i] = nodes[len(nodes)-1]
						nodes = nodes[:len(nodes)-1]
						_ = hook(nodes)
						return
					}
				}

			}

			for resp := range ch {
				for _, event := range resp.Events {
					var key = event.Kv.Key
					switch event.Type {
					case etcd.EventTypePut:
						put(key, string(event.Kv.Value))
					case etcd.EventTypeDelete:
						del(key)
					}
				}
			}
		}()
	}

	resp, err := etcd.NewKV(e.Client).Get(ctx, prefix, etcd.WithPrefix())
	if err != nil {
		xl.Warnf("get etcd failed. %s %v", prefix, err)
		return nil, err
	}
	if resp.Kvs == nil || len(resp.Kvs) == 0 {
		return []dht.NodeInfo{}, err
	}
	locker.Lock()
	defer locker.Unlock()

	nodes = []dht.NodeInfo{}
	for _, kv := range resp.Kvs {
		nodes = append(nodes, dht.NodeInfo{Host: string(kv.Value), Key: kv.Key})
	}

	return nodes, nil
}
