package main

import (
	"context"
	"encoding/json"
	"net"
	"os"
	"regexp"
	"strconv"
	"strings"
	"time"

	etcd "github.com/coreos/etcd/clientv3"

	xlog "github.com/qiniu/xlog.v1"
	jsonlog "qbox.us/http/audit/jsonlog.v3"

	"qiniu.com/argus/atserving/config"
	"qiniu.com/argus/atserving/model"
	"qiniu.com/argus/video"
	"qiniu.com/argus/video/segment"
	"qiniu.com/argus/video/vframe"
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
	DebugLevel int    `json:"debug_level"`
	Workspace  string `json:"workspace"`
	Jobs       struct {
		Mgo video.JobsInMgoConfig `json:"mgo"`
	} `json:"jobs"`
	Etcd model.ConfigEtcd `json:"etcd"`

	HTTPPort     int    `json:"http_port"`
	CallBackHost string `json:"callback_host"`

	Vframe  *vframe.VframeParams       `json:"default_vframe"`
	Segment *segment.SegmentParams     `json:"default_segment"`
	OPs     *map[string]video.OPConfig `json:"default_ops"`
	Worker  *video.WorkerConfig        `json:"default_worker"`
}

// Config ...
type Config struct {
	FileConfig

	Etcd config.ConfigValue // etcd.Config

	HTTPPort     config.ConfigValue // string
	CallBackHost config.ConfigValue // string

	STSHosts config.ConfigValue // []model.ConfigSTSHost
	SQHosts  config.ConfigValue // []string
	SQD      config.ConfigValue // []model.ConfigConsumer

	Vframe  config.ConfigValue             // video.VframeParams
	Segment config.ConfigValue             // video.SegmentParams
	OPs     config.WatchPrefixConfigValues // video.OPConfig
	Worker  config.ConfigValue             // video.WorkerConfig

	JobsInMgoHost config.ConfigValue // string

	SaveConfig config.ConfigValue // video.SaveConfig
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

	{
		conf.STSHosts = config.NewStaticEtcdValue(
			etcd.NewKV(client), model.KeySTS,
			func(bs []byte) (interface{}, error) {
				var v = []model.ConfigSTSHost{}
				err := json.Unmarshal(bs, &v)
				return v, err
			},
		)
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
		conf.SQD = config.NewStaticEtcdValue(
			etcd.NewKV(client), model.KeyNSQConsumer,
			func(bs []byte) (interface{}, error) {
				var v = []model.ConfigConsumer{}
				err := json.Unmarshal(bs, &v)
				return v, err
			},
		)
	}
	{
		conf.Vframe = config.NewMultiStaticConfigValue(
			config.NewStaticConfigValue(
				func() interface{} {
					if conf.FileConfig.Vframe == nil {
						return nil
					}
					return *conf.FileConfig.Vframe
				},
			),
			config.NewStaticEtcdValue(
				etcd.NewKV(client),
				"/ava/argus/video/vframe",
				func(bs []byte) (interface{}, error) {
					var v vframe.VframeParams
					err := json.Unmarshal(bs, &v)
					return v, err
				},
			),
		)
	}
	{
		conf.Segment = config.NewMultiStaticConfigValue(
			config.NewStaticConfigValue(
				func() interface{} {
					if conf.FileConfig.Segment == nil {
						return nil
					}
					return *conf.FileConfig.Segment
				},
			),
			config.NewStaticEtcdValue(
				etcd.NewKV(client),
				"/ava/argus/video/segment",
				func(bs []byte) (interface{}, error) {
					var v segment.SegmentParams
					err := json.Unmarshal(bs, &v)
					return v, err
				},
			),
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
		conf.Worker = config.NewMultiStaticConfigValue(
			config.NewStaticConfigValue(
				func() interface{} {
					if conf.FileConfig.Worker == nil {
						return nil
					}
					return *conf.FileConfig.Worker
				},
			),
			config.NewStaticEtcdValue(
				etcd.NewKV(client),
				"/ava/argus/video/worker",
				func(bs []byte) (interface{}, error) {
					var v video.WorkerConfig
					err := json.Unmarshal(bs, &v)
					return v, err
				},
			),
		)
	}
	{
		conf.JobsInMgoHost = config.NewMultiStaticConfigValue(
			config.NewStaticConfigValue(
				func() interface{} {
					if conf.FileConfig.Jobs.Mgo.Mgo.Host == "" {
						return nil
					}
					return conf.FileConfig.Jobs.Mgo.Mgo.Host
				}),
			config.NewStaticEtcdValue(
				etcd.NewKV(client),
				"/ava/argus/video/jobs",
				func(bs []byte) (interface{}, error) {
					var v = struct {
						MgoHost string `json:"mgo_host"`
					}{}
					err := json.Unmarshal(bs, &v)
					return v.MgoHost, err
				},
			),
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

func isAddrLocal(addr net.Addr) (string, bool) {
	if ipnet, ok := addr.(*net.IPNet); !ok {
		return "", false
	} else if ipnet.IP.To4() == nil {
		return "", false
	} else {
		ip := ipnet.IP.To4()
		if ip[0] == 10 {
			return ip.String(), true
		} else if ip[0] == 172 && ip[1] >= 16 && ip[1] <= 31 {
			return ip.String(), true
		} else if ip[0] == 192 && ip[1] == 168 {
			return ip.String(), true
		}
	}
	return "", false
}
