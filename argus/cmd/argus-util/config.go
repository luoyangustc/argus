package main

import (
	"context"
	"encoding/json"
	"os"
	"strings"
	"time"

	etcd "github.com/coreos/etcd/clientv3"

	"github.com/qiniu/db/mgoutil.v3"
	"github.com/qiniu/errors"
	"github.com/qiniu/xlog.v1"
	jsonlog "qbox.us/http/audit/jsonlog.v3"

	"qiniu.com/argus/atserving/config"
	"qiniu.com/argus/atserving/model"
	argus "qiniu.com/argus/utility"
	"qiniu.com/argus/utility/server"
)

const (
	ENV_CONFIG_HOSTS = "CONFIG_HOSTS"
	_ModeStandalone  = "standalone"
	ENV_RUN_MODE     = "RUN_MODE"
)

var (
	USE_MOCK          bool
	PORT_HTTP         string
	SERVING_GATE_HOST string
	RUN_MODE          string
	CONFIG_HOSTS      []string
)

func init() {
	if os.Getenv("USE_MOCK") == "true" {
		USE_MOCK = true
	}
	PORT_HTTP = os.Getenv("PORT_HTTP")
	SERVING_GATE_HOST = os.Getenv("SERVING_GATE_HOST")
	{
		hosts := os.Getenv("CONFIG_HOSTS")
		CONFIG_HOSTS = strings.Split(hosts, ",")
	}
	RUN_MODE = os.Getenv(ENV_RUN_MODE)
}

//----------------------------------------------------------------------------//

// FileConfig ...
type FileConfig struct {
	HTTPHost   string         `json:"http_host"`
	AuditLog   jsonlog.Config `json:"audit_log"`
	DebugLevel int            `json:"debug_level"`
	ArgGate    argus.Config   `json:"gate_server"`
	ImageGroup struct {
		Mgo *mgoutil.Config `json:"mgo"`
	} `json:"image_group"`
	Server server.Config    `json:"server"`
	Etcd   model.ConfigEtcd `json:"etcd"`
}

// Config ...
type Config struct {
	FileConfig

	HTTPPort config.ConfigValue       // string
	STSHosts config.WatchConfigValue  // []model.ConfigSTSHost
	BjrunMgo config.StaticConfigValue // mongo config
}

// InitConfig ...
func InitConfig(ctx context.Context, conf *Config) error {
	{
		conf.HTTPPort = config.NewStaticConfigValue(func() interface{} { return PORT_HTTP })
	}

	if RUN_MODE == _ModeStandalone {
		return nil
	}

	xl := xlog.FromContextSafe(ctx)
	var etcdConfig etcd.Config
	etcdConfig.Endpoints = CONFIG_HOSTS
	etcdConfig.DialTimeout = time.Duration(conf.FileConfig.Etcd.DialTimeoutMs) * time.Millisecond

	client, err := etcd.New(etcdConfig)
	if err != nil {
		xl.Fatalf("new etcd client failed. %#v %v", etcdConfig, err)
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
		conf.BjrunMgo = config.NewStaticEtcdValue(
			etcd.NewKV(client), "/ava/argus/bjrun/mgo",
			func(bs []byte) (interface{}, error) {
				if len(bs) == 0 {
					return nil, errors.New("no valid etcd bjrun mgo config")
				}
				var v mgoutil.Config
				err := json.Unmarshal(bs, &v)
				return v, err
			},
		)
	}

	return nil
}
