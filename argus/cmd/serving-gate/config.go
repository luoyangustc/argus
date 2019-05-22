package main

import (
	"context"
	"encoding/json"
	"net"
	"os"
	"strings"
	"time"

	etcd "github.com/coreos/etcd/clientv3"
	"github.com/qiniu/xlog.v1"
	jsonlog "qbox.us/http/audit/jsonlog.v3"
	"qiniu.com/argus/atserving/config"
	"qiniu.com/argus/atserving/model"
	"qiniu.com/argus/serving_gate"
)

const (
	_ModeStandalone = "standalone"
)

const (
	ENV_CONFIG_HOSTS = "CONFIG_HOSTS"
	ENV_PORT_HTTP    = "PORT_HTTP"

	ENV_RUN_MODE = "RUN_MODE"
)

var (
	CONFIG_HOSTS []string
	PORT_HTTP    string

	RUN_MODE string
)

func init() {
	{
		hosts := os.Getenv("CONFIG_HOSTS")
		CONFIG_HOSTS = strings.Split(hosts, ",")
	}
	PORT_HTTP = os.Getenv("PORT_HTTP")

	RUN_MODE = os.Getenv(ENV_RUN_MODE)
}

//----------------------------------------------------------------------------//

// FileConfig ...
type FileConfig struct {
	AuditLog   jsonlog.Config   `json:"audit_log"`
	DebugLevel int              `json:"debug_level"`
	Etcd       model.ConfigEtcd `json:"etcd"`

	// 默认逻辑
	CallBackHost string `json:"callback_host"`
}

// Config ...
type Config struct {
	FileConfig

	HTTPPort     config.ConfigValue      // string
	CallBackHost config.ConfigValue      // string
	Etcd         config.ConfigValue      // etcd.Config
	SQHosts      config.ConfigValue      // []string
	STSHosts     config.WatchConfigValue // []model.ConfigSTSHost

	LogPush            config.WatchConfigValue
	Workers            config.WatchPrefixConfigValues // model.ConfigKeyWorker + model.ConfigWorker
	AppMetadataDefault config.WatchConfigValue        // model.ConfigAppMetadata
	AppMetadatas       config.WatchPrefixConfigValues // model.ConfigAppMetadata
	AppReleases        config.WatchPrefixConfigValues // model.ConfigKeyAppRelease + model.ConfigAppRelease
}

// InitConfig ...
func InitConfig(ctx context.Context, conf *Config) error {
	xl := xlog.FromContextSafe(ctx)

	{
		conf.HTTPPort = config.NewStaticConfigValue(func() interface{} { return PORT_HTTP })
	}

	{
		conf.CallBackHost = config.NewMultiStaticConfigValue(
			config.NewStaticConfigValue(
				func() interface{} {
					if len(conf.FileConfig.CallBackHost) == 0 {
						return nil
					}
					return conf.FileConfig.CallBackHost
				},
			),
			config.NewStaticConfigValue(
				func() interface{} {
					addrs, err := net.InterfaceAddrs()
					if err != nil {
						xl.Fatalf("get net addrs failed. %v", err)
					}
					for _, addr := range addrs {
						if host, ok := isAddrLocal(addr); ok {
							return net.JoinHostPort(host, config.GetString(ctx, conf.HTTPPort, "80"))
						}
					}
					return nil
				},
			),
		)
	}

	var etcdConfig etcd.Config
	etcdConfig.Endpoints = CONFIG_HOSTS
	etcdConfig.DialTimeout = time.Duration(conf.FileConfig.Etcd.DialTimeoutMs) * time.Millisecond
	{
		conf.Etcd = config.NewStaticConfigValue(func() interface{} { return etcdConfig })
	}
	client, err := etcd.New(etcdConfig)
	if err != nil {
		xl.Fatalf("new etcd client failed. %#v %v", etcdConfig, err)
	}

	{
		conf.SQHosts = config.NewStaticEtcdValue(
			etcd.NewKV(client), model.KeyNSQProducer,
			func(bs []byte) (interface{}, error) {
				var v = []string{}
				err := json.Unmarshal(bs, &v)
				return v, err
			},
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
		conf.LogPush = config.NewWatchEtcdValue(
			ctx, client, model.KeyLogPush,
			func(v []byte) (value interface{}, err error) {
				var cfg gate.LogPushConfig
				if v == nil {
					return
				}
				if err = json.Unmarshal(v, &cfg); err != nil {
					return
				}
				value = cfg
				return
			},
		)
	}
	{
		conf.Workers = config.NewWatchPrefixEtcdValues(
			ctx, client, model.KeyWorkerPrefix,
			func(ctx context.Context, kbs, vbs []byte) (key, value interface{}, err error) {
				var (
					k model.ConfigKeyWorker
					v model.ConfigWorker
				)
				if err = (&k).Parse(kbs); err != nil {
					return
				}
				key = k
				if vbs != nil {
					if err = json.Unmarshal(vbs, &v); err != nil {
						return
					}
					value = v
				}
				return
			},
		)
	}
	{
		conf.AppMetadataDefault = config.NewWatchEtcdValue(
			ctx, client, model.KeyAppMetadataDefault,
			func(bs []byte) (interface{}, error) {
				var v = model.ConfigAppMetadata{}
				err := json.Unmarshal(bs, &v)
				return v, err
			},
		)
		conf.AppMetadatas = config.NewWatchPrefixEtcdValues(
			ctx, client, model.KeyAppMetadataPrefix,
			func(ctx context.Context, kbs, vbs []byte) (key, value interface{}, err error) {
				var (
					k model.ConfigKeyAppMetadata
					v model.ConfigAppMetadata
				)
				if err = (&k).Parse(kbs); err != nil {
					return
				}
				key = k
				if vbs != nil {
					if err = json.Unmarshal(vbs, &v); err != nil {
						return
					}
					value = v
				}
				return
			},
		)
	}
	{
		conf.AppReleases = config.NewWatchPrefixEtcdValues(
			ctx, client, model.KeyAppReleasePrefix,
			func(ctx context.Context, kbs, vbs []byte) (key, value interface{}, err error) {
				var (
					k model.ConfigKeyAppRelease
					v model.ConfigAppRelease
				)
				if err = (&k).Parse(kbs); err != nil {
					return
				}
				key = k
				if vbs != nil {
					if err = json.Unmarshal(vbs, &v); err != nil {
						return
					}
					value = v
				}
				return
			},
		)
	}

	return nil
}
