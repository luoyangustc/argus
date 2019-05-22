package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"time"

	etcd "github.com/coreos/etcd/clientv3"

	"github.com/qiniu/xlog.v1"
	jsonlog "qbox.us/http/audit/jsonlog.v3"

	"qiniu.com/argus/atserving/config"
	"qiniu.com/argus/atserving/model"
	"qiniu.com/argus/bjob/job"
	"qiniu.com/argus/bjob/workers"
)

const (
	ENV_CONFIG_HOSTS = "CONFIG_HOSTS"
	ENV_PORT_HTTP    = "PORT_HTTP"
	ENV_JOB_WORKER   = "JOB_WORKER"
)

var (
	CONFIG_HOSTS []string
	PORT_HTTP    string
	JOB_WORKER   string
)

func init() {
	{
		hosts := os.Getenv(ENV_CONFIG_HOSTS)
		CONFIG_HOSTS = strings.Split(hosts, ",")
	}
	PORT_HTTP = os.Getenv(ENV_PORT_HTTP)
	JOB_WORKER = os.Getenv(ENV_JOB_WORKER)
}

//----------------------------------------------------------------------------//

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
	Etcd       model.ConfigEtcd `json:"etcd"`
}

// Config ...
type Config struct {
	FileConfig

	HTTPPort config.ConfigValue // string
	Etcd     config.ConfigValue // etcd.Config
	Worker   config.ConfigValue // job.WorkerNodeConfig
	SQHosts  config.ConfigValue // []string

	Biz config.ConfigValue // workers.InferenceImageConfig
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
		conf.Worker = config.NewStaticEtcdValue(
			etcd.NewKV(client), "/ava/argus/jobs/workers/"+JOB_WORKER,
			func(bs []byte) (interface{}, error) {
				var v = struct {
					Worker job.WorkerNodeConfig `json:"worker"`
				}{}
				err := json.Unmarshal(bs, &v)
				return v.Worker, err
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
		conf.Biz = config.NewStaticEtcdValue(
			etcd.NewKV(client),
			fmt.Sprintf("/ava/argus/jobs/workers/%s", JOB_WORKER),
			func(bs []byte) (interface{}, error) {
				var v workers.InferenceImageConfig
				err := json.Unmarshal(bs, &v)
				return v, err
			},
		)
	}

	return nil
}
