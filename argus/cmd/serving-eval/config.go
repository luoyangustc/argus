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

const (
	_ModelLocal     = "local"
	_ModeStandalone = "standalone"
	_ModeProduction = "production"
)

const (
	_IntegrateLib    = "lib"
	_IntegrateLib2   = "lib2"
	_IntegratePython = "py"
	_IntegrateNative = "native"
	_IntegrateZMQ    = "zmq"
	_IntegrateGrpc   = "grpc"
)

// 环境变量
const (
	ENV_BOOTS_APP     = "BOOTS_APP"
	ENV_BOOTS_VERSION = "BOOTS_VERSION"
	ENV_CONFIG_HOSTS  = "CONFIG_HOSTS"
	ENV_PORT_HTTP     = "PORT_HTTP"
	ENV_RUN_MODE      = "RUN_MODE"
	ENV_USE_DEVICE    = "USE_DEVICE"
	ENV_INTEGRATE     = "INTEGRATE"
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
	Integrate  string           `json:"integrate"`
	Etcd       model.ConfigEtcd `json:"etcd"`

	// standalone
	RunMode   string                 `json:"run_mode"`
	HTTPPort  int                    `json:"http_port"`
	UseDevice string                 `json:"use_device"`
	Release   model.ConfigAppRelease `json:"release"`
	Worker    model.ConfigWorker     `json:"worker"`
	Owner     *model.Owner           `json:"owner"`
	RsHost    *string                `json:"rs_host"`
}

// Config ...
type Config struct {
	FileConfig

	Mode struct {
		UseMockEval bool
		OpenNsq     bool
		OpenHTTP    bool
		OpenEtcd    bool
	}

	Etcd config.ConfigValue // etcd.Config

	AppName    config.ConfigValue // string
	AppVersion config.ConfigValue // string
	HTTPPort   config.ConfigValue // string
	UseDevice  config.ConfigValue // string
	Integrate  config.ConfigValue // string

	AppMetadata config.StaticConfigValue // model.ConfigAppMetadata
	AppRelease  config.StaticConfigValue // model.ConfigAppRelease

	Worker config.StaticConfigValue // model.ConfigWorker

	STSHosts config.WatchConfigValue // []model.ConfigSTSHost
	SQD      config.ConfigValue      // []model.ConfigConsumer

	BatchSize config.ConfigValue // int
	RsHost    config.ConfigValue // string
	Owner     config.ConfigValue // model.Owner
}

// InitConfig ...
func InitConfig(ctx context.Context, conf *Config) error {

	var xl = xlog.FromContextSafe(ctx)

	{
		if len(conf.RunMode) == 0 {
			conf.RunMode = os.Getenv(ENV_RUN_MODE)
		}
		if conf.RunMode == "" {
			conf.RunMode = _ModeProduction
		}
		switch conf.RunMode {
		case _ModelLocal:
			conf.Mode.UseMockEval = true
			conf.Mode.OpenNsq = true
			conf.Mode.OpenHTTP = true
			conf.Mode.OpenEtcd = true
		case _ModeStandalone:
			conf.Mode.UseMockEval = false
			conf.Mode.OpenNsq = false
			conf.Mode.OpenHTTP = true
			conf.Mode.OpenEtcd = false
		case _ModeProduction:
			conf.Mode.UseMockEval = false
			conf.Mode.OpenNsq = true
			conf.Mode.OpenHTTP = true
			conf.Mode.OpenEtcd = true
		default:
			xl.Fatalf("bad run_mode %s", conf.RunMode)
		}
	}

	var (
		etcdConfig etcd.Config
		client     *etcd.Client
		err        error
	)
	if conf.Mode.OpenEtcd {
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
	if conf.RunMode != _ModeStandalone {
		conf.AppName = config.NewStaticConfigValue(
			func() interface{} { return strings.TrimPrefix(os.Getenv(ENV_BOOTS_APP), "ava-") }) // BOOTS_APP: ava-xx; APP: xx
		conf.AppVersion = config.NewStaticConfigValue(
			func() interface{} { return os.Getenv(ENV_BOOTS_VERSION) })
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
		conf.UseDevice = config.NewMultiStaticConfigValue(
			config.NewStaticConfigValue(
				func() interface{} {
					if conf.FileConfig.UseDevice == "" {
						return nil
					}
					return conf.FileConfig.UseDevice
				}),
			config.NewStaticConfigValue(
				func() interface{} { return os.Getenv(ENV_USE_DEVICE) }),
		)
	}
	{
		conf.Integrate = config.NewMultiStaticConfigValue(
			config.NewStaticConfigValue(
				func() interface{} {
					if conf.FileConfig.Integrate == "" {
						return nil
					}
					return conf.FileConfig.Integrate
				}),
			config.NewStaticConfigValue(
				func() interface{} { return os.Getenv(ENV_INTEGRATE) }),
		)
	}

	var (
		releaseFromFile = config.NewStaticConfigValue(
			func() interface{} {
				if conf.Mode.OpenEtcd {
					return nil
				}
				return conf.FileConfig.Release
			},
		)
		releaseFromEtcd = config.NewStaticConfigValue(func() interface{} { return nil })
	)

	if conf.Mode.OpenEtcd {
		releaseFromEtcd = config.NewStaticEtcdValue(
			etcd.NewKV(client),
			model.KeyAppRelease(
				config.GetString(ctx, conf.AppName, ""),
				config.GetString(ctx, conf.AppVersion, ""),
			),
			func(bs []byte) (interface{}, error) {
				var v model.ConfigAppRelease
				err := json.Unmarshal(bs, &v)
				return v, err
			},
		)
	}

	{
		conf.AppRelease = config.NewMultiStaticConfigValue(releaseFromFile, releaseFromEtcd)
		if conf.Mode.OpenEtcd {
			conf.AppMetadata = config.NewStaticEtcdValue(
				etcd.NewKV(client), model.KeyAppMetadata(config.GetString(ctx, conf.AppName, "")),
				func(bs []byte) (interface{}, error) {
					var v model.ConfigAppMetadata
					err := json.Unmarshal(bs, &v)
					return v, err
				},
			)
		} else {
			conf.AppMetadata = config.NewStaticConfigValue(func() interface{} { return nil })
		}
	}

	if conf.Mode.OpenEtcd {
		{
			conf.Worker = config.NewMultiStaticConfigValue(
				config.NewStaticEtcdValue(
					etcd.NewKV(client),
					model.KeyWorkerAppRelease(
						config.GetString(ctx, conf.AppName, ""),
						config.GetString(ctx, conf.AppVersion, ""),
					),
					func(bs []byte) (interface{}, error) {
						var v model.ConfigWorker
						err := json.Unmarshal(bs, &v)
						return v, err
					},
				),
				config.NewStaticEtcdValue(
					etcd.NewKV(client),
					model.KeyWorkerApp(config.GetString(ctx, conf.AppName, "")),
					func(bs []byte) (interface{}, error) {
						var v model.ConfigWorker
						err := json.Unmarshal(bs, &v)
						return v, err
					},
				),
				config.NewStaticEtcdValue(
					etcd.NewKV(client),
					model.KeyWorkerDefault,
					func(bs []byte) (interface{}, error) {
						var v model.ConfigWorker
						err := json.Unmarshal(bs, &v)
						return v, err
					},
				),
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
	} else {
		conf.Worker = config.NewStaticConfigValue(func() interface{} {
			return conf.FileConfig.Worker
		})
	}

	{
		conf.BatchSize = config.NewMultiStaticConfigValue(
			releaseFromFile.Find(
				func(v1 interface{}) interface{} {
					v2 := v1.(model.ConfigAppRelease)
					if v2.BatchSize == nil {
						return 0
					}
					return *v2.BatchSize
				},
			),
			conf.AppRelease.Find(
				func(v1 interface{}) interface{} {
					v2 := v1.(model.ConfigAppRelease)
					if v2.BatchSize == nil {
						return 0
					}
					return *v2.BatchSize
				},
			),
			releaseFromEtcd.Find(
				func(v1 interface{}) interface{} {
					v2 := v1.(model.ConfigAppRelease)
					if v2.BatchSize == nil {
						return 0
					}
					return *v2.BatchSize
				},
			),
		)
	}
	{
		conf.RsHost = config.NewMultiStaticConfigValue(
			config.NewStaticConfigValue(
				func() interface{} {
					if conf.FileConfig.RsHost == nil {
						return nil
					}
					return *conf.FileConfig.RsHost
				},
			),
			conf.Worker.Find(
				func(v1 interface{}) interface{} { return v1.(model.ConfigWorker).Kodo.RsHost },
			),
		)
	}
	{
		conf.Owner = config.NewMultiStaticConfigValue(
			config.NewStaticConfigValue(
				func() interface{} {
					if conf.FileConfig.Owner == nil {
						return nil
					}
					return *conf.FileConfig.Owner
				},
			),
			conf.AppMetadata.Find(
				func(v1 interface{}) interface{} { return v1.(model.ConfigAppMetadata).Owner },
			),
		)
	}

	return nil
}
