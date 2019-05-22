package main

import (
	"context"
	"encoding/json"
	"net"
	"os"
	"strings"
	"time"

	"qiniu.com/argus/tuso/client/tuso_hub"

	"qiniu.com/argus/tuso/client/image_feature"
	"qiniu.com/argus/tuso/tuso_job"

	etcd "github.com/coreos/etcd/clientv3"
	"github.com/qiniu/xlog.v1"
	jsonlog "qbox.us/http/audit/jsonlog.v3"

	"qiniu.com/argus/atserving/config"
	"qiniu.com/argus/atserving/model"
	"qiniu.com/argus/bjob/job"
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
		hosts := os.Getenv(ENV_CONFIG_HOSTS)
		CONFIG_HOSTS = strings.Split(hosts, ",")
	}
	PORT_HTTP = os.Getenv(ENV_PORT_HTTP)
}

//----------------------------------------------------------------------------//

// FileConfig ...
type FileConfig struct {
	AuditLog   jsonlog.Config   `json:"audit_log"`
	DebugLevel int              `json:"debug_level"`
	Etcd       model.ConfigEtcd `json:"etcd"`
	Cmd        string           `json:"cmd"`
	// 默认逻辑
	LocalHost string `json:"local_host"`
}

// Config ...
type Config struct {
	FileConfig

	HTTPPort  config.ConfigValue // string
	LocalHost config.ConfigValue // string
	Etcd      config.ConfigValue // etcd.Config

	SearchConfig config.ConfigValue // tuso.SearchConfig
	InternalApi  config.ConfigValue // tuso_hub.Config
	FeatureApi   config.ConfigValue // image_feature.FeatureApiConfig
	Cmd          config.ConfigValue // job.MasterNodeConfig
	MQ           config.ConfigValue // job.MgoAsMQConfig
	SQHosts      config.ConfigValue // []string
}

// InitConfig ...
func InitConfig(ctx context.Context, conf *Config) error {
	xl := xlog.FromContextSafe(ctx)

	{
		conf.HTTPPort = config.NewStaticConfigValue(func() interface{} { return PORT_HTTP })
	}
	{
		conf.LocalHost = config.NewMultiStaticConfigValue(
			config.NewStaticConfigValue(
				func() interface{} {
					if len(conf.FileConfig.LocalHost) == 0 {
						return nil
					}
					return net.JoinHostPort(conf.FileConfig.LocalHost, config.GetString(ctx, conf.HTTPPort, "80"))
				},
			),
			config.NewStaticConfigValue(
				func() interface{} {
					if os.Getenv("ENV") == "LOCAL" {
						return net.JoinHostPort("127.0.0.1", config.GetString(ctx, conf.HTTPPort, "80"))
					}

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
		conf.SearchConfig = config.NewStaticEtcdValue(
			etcd.NewKV(client), "/ava/argus/jobs/cmds/"+conf.FileConfig.Cmd+"/config",
			func(bs []byte) (interface{}, error) {
				var v = tuso_job.SearchConfig{}
				err := json.Unmarshal(bs, &v)
				return v, err
			},
		)
	}
	{
		conf.Cmd = config.NewStaticEtcdValue(
			etcd.NewKV(client), "/ava/argus/jobs/cmds/"+conf.FileConfig.Cmd,
			func(bs []byte) (interface{}, error) {
				var v = struct {
					Master job.MasterNodeConfig `json:"master"`
				}{}
				err := json.Unmarshal(bs, &v)
				return v, err
			},
		)
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
		conf.FeatureApi = config.NewStaticEtcdValue(
			etcd.NewKV(client), "/ava/argus/tuso/image_feature_api",
			func(bs []byte) (interface{}, error) {
				var v image_feature.FeatureApiConfig
				err := json.Unmarshal(bs, &v)
				return v, err
			},
		)
	}
	{
		conf.InternalApi = config.NewStaticEtcdValue(
			etcd.NewKV(client), "/ava/argus/tuso/internal_api",
			func(bs []byte) (interface{}, error) {
				var v tuso_hub.Config
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
