package main

import (
	"context"
	"encoding/json"
	"os"
	"strings"
	"time"

	"qiniu.com/argus/tuso/hub"

	"github.com/qiniu/db/mgoutil.v3"

	etcd "github.com/coreos/etcd/clientv3"
	"github.com/qiniu/xlog.v1"
	jsonlog "qbox.us/http/audit/jsonlog.v3"
	"qiniu.com/argus/atserving/config"
	"qiniu.com/argus/atserving/model"
)

const (
	ENV_CONFIG_HOSTS = "CONFIG_HOSTS"
)

var (
	CONFIG_HOSTS []string
	PORT_HTTP    string
)

func init() {
	{
		hosts := os.Getenv("CONFIG_HOSTS")
		CONFIG_HOSTS = strings.Split(hosts, ",")
	}
	PORT_HTTP = os.Getenv("PORT_HTTP")
}

//----------------------------------------------------------------------------//

// FileConfig ...
type FileConfig struct {
	AuditLog   jsonlog.Config   `json:"audit_log"`
	DebugLevel int              `json:"debug_level"`
	Etcd       model.ConfigEtcd `json:"etcd"`
}

// Config ...
type Config struct {
	FileConfig

	HTTPPort config.ConfigValue // string

	Mgo        config.StaticConfigValue `json:"mgo"`
	Kodo       config.StaticConfigValue `json:"kodo"`
	JobGateApi config.StaticConfigValue `json:"jobgate_api"`
}

// InitConfig ...
func InitConfig(ctx context.Context, conf *Config) error {
	{
		conf.HTTPPort = config.NewStaticConfigValue(func() interface{} { return PORT_HTTP })
	}

	xl := xlog.FromContextSafe(ctx)

	{
		conf.HTTPPort = config.NewStaticConfigValue(func() interface{} { return PORT_HTTP })
	}

	var etcdConfig etcd.Config
	etcdConfig.Endpoints = CONFIG_HOSTS
	etcdConfig.DialTimeout = time.Duration(conf.FileConfig.Etcd.DialTimeoutMs) * time.Millisecond

	client, err := etcd.New(etcdConfig)
	if err != nil {
		xl.Fatalf("new etcd client failed. %#v %v", etcdConfig, err)
	}

	{
		conf.Mgo = config.NewStaticEtcdValue(
			etcd.NewKV(client), "/ava/argus/tuso/mgo",
			func(bs []byte) (interface{}, error) {
				var v mgoutil.Config
				err := json.Unmarshal(bs, &v)
				return v, err
			},
		)
	}

	{
		conf.Kodo = config.NewStaticEtcdValue(
			etcd.NewKV(client), "/ava/argus/tuso/kodo",
			func(bs []byte) (interface{}, error) {
				var v hub.KodoConfig
				err := json.Unmarshal(bs, &v)
				return v, err
			},
		)
	}

	{
		conf.JobGateApi = config.NewStaticEtcdValue(
			etcd.NewKV(client), "/ava/argus/tuso/jobgate_api",
			func(bs []byte) (interface{}, error) {
				var v hub.JobGateApiConfig
				err := json.Unmarshal(bs, &v)
				return v, err
			},
		)
	}

	return nil
}
