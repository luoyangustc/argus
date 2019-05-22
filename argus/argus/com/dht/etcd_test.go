package dht

import (
	"context"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"

	etcd "github.com/coreos/etcd/clientv3"
	"github.com/stretchr/testify/assert"

	"qbox.us/dht"
)

var _ETCD_HOST = []string{"localhost:2379"}

func TestEtcd(t *testing.T) {

	var prefix = "/ava/argus/dht/TEST_" + time.Now().Format("20060102150405")

	reg := func(host string) context.CancelFunc {

		cli, err := etcd.New(
			etcd.Config{
				Endpoints:   _ETCD_HOST,
				DialTimeout: time.Second,
			},
		)
		assert.NoError(t, err)

		c := NewETCD(cli, prefix)
		ctx, cancel := context.WithCancel(context.Background())
		err = c.Register(ctx, host, 2)
		assert.NoError(t, err)
		return cancel
	}

	copyF := func(nodes dht.NodeInfos) dht.NodeInfos {
		ns := make([]dht.NodeInfo, 0, len(nodes))
		for _, node := range nodes {
			ns = append(ns, dht.NodeInfo{Key: node.Key, Host: node.Host})
		}
		return ns
	}

	_ = reg("host1")
	cancel2 := reg("host2")
	time.Sleep(time.Second)

	var nodes dht.NodeInfos
	var locker = new(sync.Mutex)
	{
		cli, err := etcd.New(
			etcd.Config{
				Endpoints:   _ETCD_HOST,
				DialTimeout: time.Second,
			},
		)
		assert.NoError(t, err)
		c := NewETCD(cli, prefix)

		nodes, err = c.Nodes(context.Background(), nil)
		assert.NoError(t, err)

		assert.Equal(t, 2, len(nodes))
		sort.Slice(nodes, func(i, j int) bool { return strings.Compare(nodes[i].Host, nodes[j].Host) <= 0 })
		for i, node := range nodes {
			assert.Equal(t, []string{"host1", "host2"}[i], node.Host)
		}
	}

	cancel3 := reg("host3")
	time.Sleep(time.Second)
	{
		cli, err := etcd.New(
			etcd.Config{
				Endpoints:   _ETCD_HOST,
				DialTimeout: time.Second,
			},
		)
		assert.NoError(t, err)
		c := NewETCD(cli, prefix)

		nodes, err = c.Nodes(context.Background(),
			func(ns dht.NodeInfos) error {
				locker.Lock()
				defer locker.Unlock()
				nodes = copyF(ns)
				return nil
			})
		assert.NoError(t, err)

		func() {
			locker.Lock()
			defer locker.Unlock()

			assert.Equal(t, 3, len(nodes))
			nodes = copyF(nodes)
			sort.Slice(nodes, func(i, j int) bool { return strings.Compare(nodes[i].Host, nodes[j].Host) <= 0 })
			for i, node := range nodes {
				assert.Equal(t, []string{"host1", "host2", "host3"}[i], node.Host)
			}
		}()
	}

	_ = reg("host4")
	time.Sleep(time.Second * 3)
	func() {
		locker.Lock()
		defer locker.Unlock()

		t.Logf("%#v", nodes)
		assert.Equal(t, 4, len(nodes))
		sort.Slice(nodes, func(i, j int) bool { return strings.Compare(nodes[i].Host, nodes[j].Host) <= 0 })
		for i, node := range nodes {
			assert.Equal(t, []string{"host1", "host2", "host3", "host4"}[i], node.Host)
		}
	}()

	cancel2()
	cancel3()
	time.Sleep(time.Second * 4)
	func() {
		locker.Lock()
		defer locker.Unlock()

		assert.Equal(t, 2, len(nodes))
		sort.Slice(nodes, func(i, j int) bool { return strings.Compare(nodes[i].Host, nodes[j].Host) <= 0 })
		for i, node := range nodes {
			assert.Equal(t, []string{"host1", "host4"}[i], node.Host)
		}
	}()

}
