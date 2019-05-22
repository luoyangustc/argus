package main

import (
	"context"
	"encoding/json"
	"os"
	"strconv"
	"strings"
	"time"

	etcd "github.com/coreos/etcd/clientv3"

	"github.com/qiniu/db/mgoutil.v3"
	"github.com/qiniu/xlog.v1"
	jsonlog "qbox.us/http/audit/jsonlog.v3"

	"qiniu.com/argus/atserving/config"
	"qiniu.com/argus/atserving/model"
	"qiniu.com/argus/dbstorage/proto"
)

const (
	ENV_CONFIG_HOSTS = "CONFIG_HOSTS"
	ENV_PORT_HTTP    = "PORT_HTTP"
)

var (
	PORT_HTTP    string
	CONFIG_HOSTS []string
)

func init() {
	PORT_HTTP = os.Getenv("PORT_HTTP")
	hosts := os.Getenv(ENV_CONFIG_HOSTS)
	CONFIG_HOSTS = strings.Split(hosts, ",")
}

type FileConfig struct {
	HTTPPort            int                      `json:"http_port"`
	AuditLog            jsonlog.Config           `json:"audit_log"`
	DebugLevel          int                      `json:"debug_level"`
	Etcd                model.ConfigEtcd         `json:"etcd"`
	Mgo                 *mgoutil.Config          `json:"mgo"`
	FeatureGroupService *proto.BaseServiceConfig `json:"feature_group_service"`
	ServingService      *proto.BaseServiceConfig `json:"serving_service"`
	ThreadNum           int                      `json:"thread_num"`
	MaxParallelTaskNum  int                      `json:"max_parallel_task_num"`
	IsPrivate           bool                     `json:"is_private"`
}

type Config struct {
	FileConfig
	HTTPPort            config.ConfigValue // string
	Mgo                 config.StaticConfigValue
	FeatureGroupService config.StaticConfigValue // proto.BaseServiceConfig
	ServingService      config.StaticConfigValue // proto.BaseServiceConfig
	ThreadNum           config.StaticConfigValue // int
	MaxParallelTaskNum  config.StaticConfigValue // int
}

// InitConfig ...
func InitConfig(ctx context.Context, conf *Config) error {
	xl := xlog.FromContextSafe(ctx)
	var etcdConfig etcd.Config
	etcdConfig.Endpoints = CONFIG_HOSTS
	etcdConfig.DialTimeout = time.Duration(conf.FileConfig.Etcd.DialTimeoutMs) * time.Millisecond

	client, err := etcd.New(etcdConfig)
	if err != nil {
		xl.Fatalf("new etcd client failed. %#v %v", etcdConfig, err)
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
		conf.Mgo = config.NewMultiStaticConfigValue(
			config.NewStaticConfigValue(
				func() interface{} {
					if conf.FileConfig.Mgo != nil {
						return *conf.FileConfig.Mgo
					}
					return nil
				}),
			config.NewStaticEtcdValue(
				etcd.NewKV(client), "/ava/argus/dbstorage/mgo",
				func(bs []byte) (interface{}, error) {
					var v mgoutil.Config
					err := json.Unmarshal(bs, &v)
					return v, err
				}),
		)
	}
	{
		conf.FeatureGroupService = config.NewMultiStaticConfigValue(
			config.NewStaticConfigValue(
				func() interface{} {
					if conf.FileConfig.FeatureGroupService != nil {
						return *conf.FileConfig.FeatureGroupService
					}
					return nil
				}),
			config.NewStaticEtcdValue(
				etcd.NewKV(client), "/ava/argus/dbstorage/feature_group",
				func(bs []byte) (interface{}, error) {
					var v proto.BaseServiceConfig
					err := json.Unmarshal(bs, &v)
					return v, err
				}),
		)
	}
	{
		conf.ServingService = config.NewMultiStaticConfigValue(
			config.NewStaticConfigValue(
				func() interface{} {
					if conf.FileConfig.ServingService != nil {
						return *conf.FileConfig.ServingService
					}
					return nil
				}),
			config.NewStaticEtcdValue(
				etcd.NewKV(client), "/ava/argus/dbstorage/serving",
				func(bs []byte) (interface{}, error) {
					var v proto.BaseServiceConfig
					err := json.Unmarshal(bs, &v)
					return v, err
				}),
		)
	}
	{
		conf.ThreadNum = config.NewMultiStaticConfigValue(
			config.NewStaticConfigValue(
				func() interface{} {
					if conf.FileConfig.ThreadNum != 0 {
						return conf.FileConfig.ThreadNum
					}
					return nil
				}),
			config.NewStaticEtcdValue(
				etcd.NewKV(client), "/ava/argus/dbstorage/thread_num",
				func(bs []byte) (interface{}, error) {
					var v int
					err := json.Unmarshal(bs, &v)
					return v, err
				}),
		)
	}
	{
		conf.MaxParallelTaskNum = config.NewMultiStaticConfigValue(
			config.NewStaticConfigValue(
				func() interface{} {
					if conf.FileConfig.MaxParallelTaskNum != 0 {
						return conf.FileConfig.MaxParallelTaskNum
					}
					return nil
				}),
			config.NewStaticEtcdValue(
				etcd.NewKV(client), "/ava/argus/dbstorage/max_run_task",
				func(bs []byte) (interface{}, error) {
					var v int
					err := json.Unmarshal(bs, &v)
					return v, err
				}),
		)
	}
	return nil
}
