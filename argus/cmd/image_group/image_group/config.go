package main

import (
	"context"
	"encoding/json"
	"os"
	"regexp"
	"strings"
	"time"

	etcd "github.com/coreos/etcd/clientv3"

	"github.com/qiniu/db/mgoutil.v3"
	"github.com/qiniu/xlog.v1"
	jsonlog "qbox.us/http/audit/jsonlog.v3"

	"qiniu.com/argus/atserving/config"
	"qiniu.com/argus/atserving/model"
	IG "qiniu.com/argus/feature_group/imageg"
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

	Mgo    *mgoutil.Config `json:"mgo"`
	Search *IG.Config      `json:"search"`
}

// Config ...
type Config struct {
	FileConfig

	HTTPPort config.ConfigValue // string

	STSHosts config.WatchConfigValue // []model.ConfigSTSHost

	Mgo            config.StaticConfigValue
	Search         config.StaticConfigValue
	FeatureAPIs    config.WatchPrefixConfigValues // imageg.ImageGFeatureAPIConfig
	FeatureVersion config.WatchConfigValue        // string
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
		conf.Mgo = config.NewMultiStaticConfigValue(
			config.NewStaticConfigValue(
				func() interface{} {
					if conf.FileConfig.Mgo != nil {
						return *conf.FileConfig.Mgo
					}
					return nil
				}),
			config.NewStaticEtcdValue(
				etcd.NewKV(client), "/ava/argus/image_group/mgo",
				func(bs []byte) (interface{}, error) {
					var v mgoutil.Config
					err := json.Unmarshal(bs, &v)
					return v, err
				},
			),
		)
	}
	{
		conf.Search = config.NewMultiStaticConfigValue(
			config.NewStaticConfigValue(
				func() interface{} {
					if conf.FileConfig.Search != nil {
						return *conf.FileConfig.Search
					}
					return nil
				}),
			config.NewStaticEtcdValue(
				etcd.NewKV(client), "/ava/argus/image_group/search",
				func(bs []byte) (interface{}, error) {
					var v IG.Config
					err := json.Unmarshal(bs, &v)
					return v, err
				},
			),
		)
	}
	{
		var (
			prefix          = "/ava/argus/image_group/feature"
			reFeatureConfig = regexp.MustCompile(prefix + "/([^/]+)")
		)
		conf.FeatureAPIs = config.NewWatchPrefixEtcdValues(
			ctx, client, prefix,
			func(ctx context.Context, kbs, vbs []byte) (key, value interface{}, err error) {
				var (
					v IG.ImageGFeatureAPIConfig
				)
				if rest := reFeatureConfig.FindStringSubmatch(string(kbs)); rest != nil {
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
		conf.FeatureVersion = config.NewWatchEtcdValue(
			ctx, client, "/ava/argus/image_group/current_feature",
			func(v []byte) (value interface{}, err error) {
				var version string
				if v == nil {
					return
				}
				if err = json.Unmarshal(v, &version); err != nil {
					return
				}
				value = version
				return
			},
		)
	}

	return nil
}
