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
	FG "qiniu.com/argus/feature_group"
)

const (
	ENV_CONFIG_HOSTS = "CONFIG_HOSTS"
)

var (
	CONFIG_HOSTS []string
	PORT_HTTP    string

	client *etcd.Client
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

	Storage *FG.StorageConfig `json:"storage"`
	Fetch   string            `json:"fetch"`

	// 默认逻辑
	LocalHost string `json:"local_host"`
}

// Config ...
type Config struct {
	FileConfig

	HTTPPort  config.ConfigValue // string
	LocalHost config.ConfigValue // string

	Storage config.StaticConfigValue
	Fetch   config.StaticConfigValue
}

// InitConfig ...
func InitConfig(ctx context.Context, conf *Config) error {
	{
		conf.HTTPPort = config.NewStaticConfigValue(func() interface{} { return PORT_HTTP })
	}

	xl := xlog.FromContextSafe(ctx)

	var (
		etcdConfig etcd.Config
		err        error
	)
	etcdConfig.Endpoints = CONFIG_HOSTS
	etcdConfig.DialTimeout = time.Duration(conf.FileConfig.Etcd.DialTimeoutMs) * time.Millisecond

	client, err = etcd.New(etcdConfig)
	if err != nil {
		xl.Fatalf("new etcd client failed. %#v %v", etcdConfig, err)
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
	{
		conf.Storage = config.NewMultiStaticConfigValue(
			config.NewStaticConfigValue(
				func() interface{} {
					if conf.FileConfig.Storage != nil {
						return *conf.FileConfig.Storage
					}
					return nil
				}),
			config.NewStaticEtcdValue(
				etcd.NewKV(client), "/ava/argus/face_group/cache",
				func(bs []byte) (interface{}, error) {
					var v FG.StorageConfig
					err := json.Unmarshal(bs, &v)
					return v, err
				},
			),
		)
	}
	{
		conf.Fetch = config.NewMultiStaticConfigValue(
			config.NewStaticConfigValue(
				func() interface{} {

					if len(conf.FileConfig.Fetch) == 0 {
						return nil
					}
					return struct {
						URL string `json:"url"`
					}{URL: conf.FileConfig.Fetch}
				}),
			config.NewStaticEtcdValue(
				etcd.NewKV(client), "/ava/argus/face_group/fetch",
				func(bs []byte) (interface{}, error) {
					var v struct {
						URL string `json:"url"`
					}
					err := json.Unmarshal(bs, &v)
					return v, err
				},
			),
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
