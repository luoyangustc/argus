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
	"qiniu.com/argus/tuso/hub"
)

const (
	ENV_CONFIG_HOSTS = "CONFIG_HOSTS"
	ENV_PORT_HTTP    = "PORT_HTTP"
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
	Cmd        string           `json:"cmd"`
}

// Config ...
type Config struct {
	FileConfig

	HTTPPort config.ConfigValue // string
	Etcd     config.ConfigValue // etcd.Config
	Cmd      config.ConfigValue // job.WorkerNodeConfig
	SQHosts  config.ConfigValue // []string
	Kodo     config.ConfigValue // hub.KodoConfig
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
		conf.Cmd = config.NewStaticEtcdValue(
			etcd.NewKV(client), "/ava/argus/jobs/cmds/"+conf.FileConfig.Cmd,
			func(bs []byte) (interface{}, error) {
				var v = struct {
					Master job.WorkerNodeConfig `json:"worker"`
				}{}
				err := json.Unmarshal(bs, &v)
				return v, err
			},
		)
	}

	{
		conf.SQHosts = config.NewStaticEtcdValue(
			etcd.NewKV(client), model.KeyNSQConsumer,
			func(bs []byte) (interface{}, error) {
				var v = []model.ConfigConsumer{}
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

	return nil
}
