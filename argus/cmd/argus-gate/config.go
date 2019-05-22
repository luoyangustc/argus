package main

import (
	"context"
	"encoding/json"
	"errors"
	"io/ioutil"
	"os"
	"strconv"
	"strings"
	"time"

	etcd "github.com/coreos/etcd/clientv3"

	xlog "github.com/qiniu/xlog.v1"
	jsonlog "qbox.us/http/audit/jsonlog.v3"

	"qiniu.com/argus/argus/gate"
	"qiniu.com/argus/atserving/config"
	"qiniu.com/argus/atserving/model"
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

// FileConfig ...
type FileConfig struct {
	AuditLog   jsonlog.Config   `json:"audit_log"`
	DebugLevel int              `json:"debug_level"`
	Etcd       model.ConfigEtcd `json:"etcd"`

	// standalone
	HTTPPort    int               `json:"http_port"`
	ProxyRoutes []gate.ProxyRoute `json:"proxy"`
}

// Config ...
type Config struct {
	FileConfig

	Etcd     config.ConfigValue // etcd.Config
	HTTPPort config.ConfigValue // string

	ProxyRoutes config.WatchPrefixConfigValues // gate.ProxyRoute
	FetchValue  func(context.Context, string) ([]byte, error)
}

// InitConfig ...
func InitConfig(ctx context.Context, conf *Config) error {

	var (
		xl         = xlog.FromContextSafe(ctx)
		etcdConfig etcd.Config
		client     *etcd.Client
		err        error
	)

	if RUN_MODE != _ModeStandalone {
		etcdConfig.Endpoints = CONFIG_HOSTS
		etcdConfig.DialTimeout = time.Duration(conf.FileConfig.Etcd.DialTimeoutMs) * time.Millisecond
		{
			conf.Etcd = config.NewStaticConfigValue(func() interface{} { return etcdConfig })
		}
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
				func() interface{} { return PORT_HTTP }),
		)
	}
	{
		conf.ProxyRoutes = config.NewWatchPrefixEtcdValues(
			ctx, client,
			"/ava/argus/gate/routes",
			func(ctx context.Context, kbs, vbs []byte) (key, value interface{}, err error) {
				var (
					v gate.ProxyRoute
				)
				xl.Infof("%s %s", string(kbs), string(vbs))
				key = strings.TrimPrefix(string(kbs), "/ava/argus/gate/routes")
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

	if RUN_MODE == _ModeStandalone {
		conf.FetchValue = func(ctx context.Context, file string) ([]byte, error) {
			return ioutil.ReadFile(file)
		}
	} else {
		conf.FetchValue = func(ctx context.Context, key string) ([]byte, error) {
			var (
				xl     = xlog.FromContextSafe(ctx)
				resp   *etcd.GetResponse
				client = etcd.NewKV(client)
			)
			resp, err := client.Get(ctx, "/ava/argus/gate/tls"+key)
			if err != nil {
				xl.Warnf("get etcd failed. %s %v", key, err)
				return nil, err
			}
			if resp.Kvs == nil || len(resp.Kvs) == 0 {
				xl.Warnf("get etcd failed. %s %#v", key, resp)
				return nil, errors.New("get etcd failed")
			}
			if resp.Kvs[0].Value == nil || len(resp.Kvs[0].Value) == 0 {
				xl.Warnf("got invalid etcd value")
				return nil, errors.New("got invalid etcd value")
			}
			return resp.Kvs[0].Value, nil
		}
	}

	return nil
}
