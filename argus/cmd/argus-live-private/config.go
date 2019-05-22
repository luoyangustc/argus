package argus_live

import (
	"context"
	"os"

	jsonlog "qbox.us/http/audit/jsonlog.v3"

	"qiniu.com/argus/atserving/config"
	"qiniu.com/argus/video"
	"qiniu.com/argus/video/segment"
	"qiniu.com/argus/video/vframe"
)

// 环境变量
const (
	// ENV_CONFIG_HOSTS = "CONFIG_HOSTS"
	ENV_PORT_HTTP = "PORT_HTTP"
)

// FileConfig ...
type FileConfig struct {
	AuditLog   jsonlog.Config `json:"audit_log"`
	DebugLevel int            `json:"debug_level"`
	Workspace  string         `json:"workspace"`

	HTTPPort string `json:"http_port"`
	// Host     string `json:"host"`

	Jobs struct {
		Mgo video.JobsInMgoConfig `json:"mgo"`
	} `json:"jobs"`
	Vframe      vframe.VframeParams       `json:"default_vframe"`
	Segment     segment.SegmentParams     `json:"default_segment"`
	LiveTimeout float64                   `json:"default_live_timeout"`
	OPs         map[string]video.OPConfig `json:"default_ops"`
	Worker      *video.WorkerConfig       `json:"default_worker"`
	SaveConfig  video.FileSaveConfig      `json:"default_save"`
}

// Config ...
type Config struct {
	FileConfig

	HTTPPort      config.ConfigValue // string
	JobsInMgoHost config.ConfigValue // string
	Worker        config.ConfigValue // video.WorkerConfig
}

// InitConfig ...
func InitConfig(ctx context.Context, conf *Config) error {

	{
		conf.HTTPPort = config.NewMultiStaticConfigValue(
			config.NewStaticConfigValue(
				func() interface{} {
					if conf.FileConfig.HTTPPort == "" {
						return nil
					}
					return conf.FileConfig.HTTPPort
				}),
			config.NewStaticConfigValue(
				func() interface{} { return os.Getenv(ENV_PORT_HTTP) }),
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
		)
	}

	return nil
}
