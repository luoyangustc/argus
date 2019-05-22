package main

import (
	"context"
	"encoding/json"
	"os"
	"regexp"
	"strings"
	"time"

	etcd "github.com/coreos/etcd/clientv3"

	"github.com/qiniu/xlog.v1"
	jsonlog "qbox.us/http/audit/jsonlog.v3"

	"qiniu.com/argus/atserving/config"
	"qiniu.com/argus/atserving/model"
	"qiniu.com/argus/bjob/job"
	"qiniu.com/argus/video"
	"qiniu.com/argus/video/segment"
	"qiniu.com/argus/video/vframe"
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
	Workspace  string           `json:"workspace"`
	Etcd       model.ConfigEtcd `json:"etcd"`
}

// Config ...
type Config struct {
	FileConfig

	HTTPPort config.ConfigValue // string
	Etcd     config.ConfigValue // etcd.Config
	Worker   config.ConfigValue // job.WorkerNodeConfig
	SQHosts  config.ConfigValue // []string

	STSHosts   config.WatchConfigValue        // []model.ConfigSTSHost
	Vframe     config.ConfigValue             // video.VframeParams
	Segment    config.ConfigValue             // video.SegmentParams
	OPs        config.WatchPrefixConfigValues // video.OPConfig
	SaveConfig config.ConfigValue             // video.SaveConfig
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
		conf.STSHosts = config.NewWatchEtcdValue(
			ctx, client, model.KeySTS,
			func(bs []byte) (interface{}, error) {
				var v = []model.ConfigSTSHost{}
				err := json.Unmarshal(bs, &v)
				return v, err
			},
		)
	}

	// 以下配置同argus-video
	{
		conf.Vframe = config.NewStaticEtcdValue(
			etcd.NewKV(client),
			"/ava/argus/video/vframe",
			func(bs []byte) (interface{}, error) {
				var v vframe.VframeParams
				err := json.Unmarshal(bs, &v)
				return v, err
			},
		)
	}
	{
		conf.Segment = config.NewStaticEtcdValue(
			etcd.NewKV(client),
			"/ava/argus/video/segment",
			func(bs []byte) (interface{}, error) {
				var v segment.SegmentParams
				err := json.Unmarshal(bs, &v)
				return v, err
			},
		)
	}
	{
		var (
			prefix     = "/ava/argus/video/op/config"
			reOPConfig = regexp.MustCompile(prefix + "/([^/]+)")
		)
		conf.OPs = config.NewWatchPrefixEtcdValues(
			ctx, client, prefix,
			func(ctx context.Context, kbs, vbs []byte) (key, value interface{}, err error) {
				var (
					v video.OPConfig
				)
				if rest := reOPConfig.FindStringSubmatch(string(kbs)); rest != nil {
					key = rest[1]
				} else {
					err = model.ErrBadConfigKey
					return
				}
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
		conf.SaveConfig = config.NewStaticEtcdValue(
			etcd.NewKV(client),
			"/ava/argus/video/save",
			func(bs []byte) (interface{}, error) {
				var v video.KodoSaveConfig
				err := json.Unmarshal(bs, &v)
				return v, err
			},
		)
	}

	return nil
}
