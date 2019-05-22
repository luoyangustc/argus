package main

import (
	"context"
	"encoding/json"
	"os"
	"strconv"
	"strings"
	"time"

	"qiniupkg.com/api.v7/kodo"

	etcd "github.com/coreos/etcd/clientv3"

	xlog "github.com/qiniu/xlog.v1"
	jsonlog "qbox.us/http/audit/jsonlog.v3"
	"qbox.us/qconf/qconfapi"

	"qiniu.com/argus/atserving/config"
	"qiniu.com/argus/atserving/model"
	"qiniu.com/argus/ccp/conf"
	"qiniu.com/argus/ccp/manual/dao"
	capModel "qiniu.com/argus/ccp/manual/model"
)

// 环境变量
const (
	ENV_CONFIG_HOSTS = "CONFIG_HOSTS"
	ENV_PORT_HTTP    = "PORT_HTTP"
)

type EnvConfig struct {
	Mgo                     *dao.CcpCapMgoConfig          `json:"mgo"`
	Qconf                   *qconfapi.Config              `json:"qconf"`
	Kodo                    *kodo.Config                  `json:"kodo"`
	DomainApiHost           string                        `json:"domain_api_host"`
	CapConf                 *capModel.CAPConfig           `json:"cap_client"`
	BatchEntryProcessorConf *conf.BatchEntryProcessorConf `json:"batch_entry_processor_config"`
	BatchEntryResultConf    *conf.BatchEntryResultConf    `json:"batch_entry_result_config"`
}

// FileConfig ...
type FileConfig struct {
	AuditLog   jsonlog.Config   `json:"audit_log"`
	DebugLevel int              `json:"debug_level"`
	Etcd       model.ConfigEtcd `json:"etcd"`
	HTTPPort   int              `json:"http_port"`
	Env        *EnvConfig       `json:"env"`
}

// Config ...
type Config struct {
	FileConfig
	Etcd config.ConfigValue

	HTTPPort config.ConfigValue // string

	Env config.StaticConfigValue // EnvConfig
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
		conf.Env = config.NewMultiStaticConfigValue(
			config.NewStaticConfigValue(
				func() interface{} {
					if conf.FileConfig.Env != nil {
						return *conf.FileConfig.Env
					}
					return nil
				}),
			config.NewStaticEtcdValue(
				etcd.NewKV(client), "/ava/argus/ccp/manual",
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
