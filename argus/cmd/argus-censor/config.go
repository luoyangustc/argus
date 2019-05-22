package main

import (
	"context"
	"encoding/json"
	"os"
	"strings"
	"time"

	etcd "github.com/coreos/etcd/clientv3"

	"github.com/qiniu/xlog.v1"
	jsonlog "qbox.us/http/audit/jsonlog.v3"

	"qiniu.com/argus/atserving/config"
	"qiniu.com/argus/atserving/model"
	"qiniu.com/argus/censor"
)

const (
	ENV_CONFIG_HOSTS = "CONFIG_HOSTS"
)

var (
	PORT_HTTP    string
	CONFIG_HOSTS []string
)

func init() {
	PORT_HTTP = os.Getenv("PORT_HTTP")
	{
		hosts := os.Getenv(ENV_CONFIG_HOSTS)
		CONFIG_HOSTS = strings.Split(hosts, ",")
	}
}

//----------------------------------------------------------------------------//

type ServiceConfig struct {
	ArgusImage struct {
		Host          string        `json:"host"`
		TimeoutSecond time.Duration `json:"timeout_second"`
	} `json:"argus_image"`
	ArgusVideo struct {
		Host          string        `json:"host"`
		TimeoutSecond time.Duration `json:"timeout_second"`
	} `json:"argus_video"`
	Video censor.VideoConfig `json:"video"`
}

// FileConfig ...
type FileConfig struct {
	HTTPHost   string           `json:"http_host"`
	AuditLog   jsonlog.Config   `json:"audit_log"`
	DebugLevel int              `json:"debug_level"`
	Service    *ServiceConfig   `json:"serice"`
	Etcd       model.ConfigEtcd `json:"etcd"`
}

// Config ...
type Config struct {
	FileConfig

	HTTPPort config.ConfigValue // string
	Service  config.ConfigValue // ServiceConfig
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
		conf.Service = config.NewMultiStaticConfigValue(
			config.NewStaticConfigValue(
				func() interface{} {
					if conf.FileConfig.Service != nil {
						return *conf.FileConfig.Service
					}
					return nil
				}),
			config.NewStaticEtcdValue(
				etcd.NewKV(client), "/ava/argus/censor",
				func(bs []byte) (interface{}, error) {
					var v ServiceConfig
					err := json.Unmarshal(bs, &v)
					return v, err
				},
			),
		)
	}

	return nil
}
