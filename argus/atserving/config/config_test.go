package config

import (
	"context"
	"testing"
	"time"

	etcd "github.com/coreos/etcd/clientv3"
	"github.com/stretchr/testify/assert"
)

const _ETCD_PREFIX = "/tmp/qiniu.com/argus/atserving/model"

var _ETCD_HOST = []string{"localhost:2379"}

func cleanETCD(t *testing.T, key ...string) {
	cli, err := etcd.New(
		etcd.Config{
			Endpoints:   _ETCD_HOST,
			DialTimeout: time.Second,
		},
	)
	assert.NoError(t, err)
	for _, _key := range key {
		_, err = cli.Delete(context.Background(), _key)
		assert.NoError(t, err)
	}

}
