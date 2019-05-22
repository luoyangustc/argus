package main

import (
	"context"
	"encoding/json"
	"os"
	"strings"
	"time"

	etcd "github.com/coreos/etcd/clientv3"
	"qiniupkg.com/api.v7/kodo"

	"github.com/qiniu/db/mgoutil.v3"
	"github.com/qiniu/xlog.v1"
	jsonlog "qbox.us/http/audit/jsonlog.v3"
	"qbox.us/qconf/qconfapi"

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
		hosts := os.Getenv(ENV_CONFIG_HOSTS)
		CONFIG_HOSTS = strings.Split(hosts, ",")
	}
	PORT_HTTP = os.Getenv("PORT_HTTP")
}

type EnvConfig struct {
	Mgo           *mgoutil.Config  `json:"mgo"`
	Qconf         *qconfapi.Config `json:"qconf"`
	Kodo          *kodo.Config     `json:"kodo"`
	DomainApiHost string           `json:"domain_api_host"`
}

//----------------------------------------------------------------------------//

// FileConfig ...
type FileConfig struct {
	AuditLog   jsonlog.Config   `json:"audit_log"`
	DebugLevel int              `json:"debug_level"`
	Etcd       model.ConfigEtcd `json:"etcd"`

	Env *EnvConfig `json:"env"`
}

// Config ...
type Config struct {
	FileConfig

	HTTPPort config.ConfigValue // string

	Env config.StaticConfigValue // EnvConfig
}

// InitConfig ...
func InitConfig(ctx context.Context, conf *Config) error {
	{
		conf.HTTPPort = config.NewStaticConfigValue(func() interface{} { return PORT_HTTP })
	}

	xl := xlog.FromContextSafe(ctx)

	var etcdConfig etcd.Config
	etcdConfig.Endpoints = CONFIG_HOSTS
	etcdConfig.DialTimeout = time.Duration(conf.FileConfig.Etcd.DialTimeoutMs) * time.Millisecond

	client, err := etcd.New(etcdConfig)
	if err != nil {
		xl.Fatalf("new etcd client failed. %#v %v", etcdConfig, err)
	}

	{
		conf.Env = config.NewMultiStaticConfigValue(
			config.NewStaticConfigValue(
				func() interface{} {
					if conf.FileConfig.Env != nil {
						return *conf.FileConfig.Env
					}
					return nil
				}),
			config.NewStaticEtcdValue(
				etcd.NewKV(client), "/ava/argus/ccp/review",
				func(bs []byte) (interface{}, error) {
					var v EnvConfig
					err := json.Unmarshal(bs, &v)
					return v, err
				},
			),
		)
	}

	return nil
}
