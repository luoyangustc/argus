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
	"qiniu.com/argus/bjob/job"
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
}

// Config ...
type Config struct {
	FileConfig

	HTTPPort config.ConfigValue // string
	Etcd     config.ConfigValue // etcd.Config

	MQ   config.ConfigValue // job.MgoAsMQConfig
	Cmds config.WatchPrefixConfigValues
}

// InitConfig ...
func InitConfig(ctx context.Context, conf *Config) error {
	xl := xlog.FromContextSafe(ctx)

	{
		conf.HTTPPort = config.NewStaticConfigValue(func() interface{} { return PORT_HTTP })
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
		conf.MQ = config.NewStaticEtcdValue(
			etcd.NewKV(client), "/ava/argus/jobs/mq",
			func(bs []byte) (interface{}, error) {
				var v = job.MgoAsMQConfig{}
				err := json.Unmarshal(bs, &v)
				return v, err
			},
		)
	}
	{
		prefix := "/ava/argus/jobs/cmds/"
		conf.Cmds = config.NewWatchPrefixEtcdValues(
			ctx, client, prefix,
			func(ctx context.Context, kbs, vbs []byte) (key, value interface{}, err error) {
				var (
					v = struct {
						Cmd string `json:"cmd"`
					}{}
				)
				key = strings.SplitN(strings.TrimPrefix(string(kbs), prefix), "/", 2)[0]
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
