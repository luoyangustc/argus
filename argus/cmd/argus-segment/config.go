package main

import (
	"context"
	"encoding/json"
	"os"
	"strconv"
	"strings"
	"time"

	etcd "github.com/coreos/etcd/clientv3"

	xlog "github.com/qiniu/xlog.v1"
	jsonlog "qbox.us/http/audit/jsonlog.v3"

	"qiniu.com/argus/atserving/config"
	"qiniu.com/argus/atserving/model"
)

// 环境变量
const (
	ENV_CONFIG_HOSTS = "CONFIG_HOSTS"
	ENV_PORT_HTTP    = "PORT_HTTP"
)

// FileConfig ...
type FileConfig struct {
	AuditLog jsonlog.Config `json:"audit_log"`
	SQLog    struct {
		LogDir    string `json:"logdir"`
		LogPrefix string `json:"logprefix"`
		TimeMode  int64  `json:"timemode"`
		ChunkBits uint   `json:"chunkbits"`
	} `json:"sq_log"`
	DebugLevel int              `json:"debug_level"`
	Workspace  string           `json:"workspace"`
	Etcd       model.ConfigEtcd `json:"etcd"`

	// standalone
	HTTPPort int `json:"http_port"`
}

// Config ...
type Config struct {
	FileConfig

	Etcd config.ConfigValue // etcd.Config

	HTTPPort config.ConfigValue // string

	STSHosts config.WatchConfigValue // []model.ConfigSTSHost
	SQD      config.ConfigValue      // []model.ConfigConsumer
}

// InitConfig ...
func InitConfig(ctx context.Context, conf *Config) error {

	var xl = xlog.FromContextSafe(ctx)

	var (
		etcdConfig etcd.Config
		client     *etcd.Client
		err        error
	)
	{
		configHosts := os.Getenv(ENV_CONFIG_HOSTS)
		if configHosts == "" {
			xl.Fatal("ENV CONFIG_HOSTS required")
		}
		etcdConfig.Endpoints = strings.Split(configHosts, ",")
		etcdConfig.DialTimeout = time.Duration(conf.FileConfig.Etcd.DialTimeoutMs) * time.Millisecond
		conf.Etcd = config.NewStaticConfigValue(func() interface{} { return etcdConfig })
		client, err = etcd.New(etcdConfig)
		if err != nil {
			xl.Fatalf("new etcd client failed. %#v %v", etcdConfig, err)
		}
	}
	{
		conf.HTTPPort = config.NewMultiStaticConfigValue(
			config.NewStaticConfigValue(
				func() interface{} {
					if conf.FileConfig.HTTPPort == 0 {
						return nil
					}
					return strconv.Itoa(conf.FileConfig.HTTPPort)
				}),
			config.NewStaticConfigValue(
				func() interface{} { return os.Getenv(ENV_PORT_HTTP) }),
		)
	}

	{
		conf.STSHosts = config.NewWatchEtcdValue(
			ctx, client, model.KeySTS,
			func(bs []byte) (interface{}, error) {
				var v = []model.ConfigSTSHost{}
				err := json.Unmarshal(bs, &v)
				return v, err
			},
		)
	}
	{
		conf.SQD = config.NewStaticEtcdValue(
			etcd.NewKV(client), model.KeyNSQConsumer,
			func(bs []byte) (interface{}, error) {
				var v = []model.ConfigConsumer{}
				err := json.Unmarshal(bs, &v)
				return v, err
			},
		)
	}

	return nil
}
