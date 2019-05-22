package main

import (
	"context"
	"net"
	"os"
	"strconv"

	jsonlog "qbox.us/http/audit/jsonlog.v3"

	"qiniu.com/argus/atserving/config"
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
	Savespace  string `json:"savespace"`

	HTTPPort int    `json:"http_port"`
	Host     string `json:"host"`

	Vframe  *vframe.VframeParams       `json:"default_vframe"`
	Segment *segment.SegmentParams     `json:"default_segment"`
	OPs     *map[string]video.OPConfig `json:"default_ops"`

	HTTPHost string `json:"http_host"`
}

// Config ...
type Config struct {
	FileConfig

	HTTPPort config.ConfigValue // string
	Host     config.ConfigValue // string

	Vframe  config.ConfigValue // video.VframeParams
	Segment config.ConfigValue // video.SegmentParams
	OPs     config.ConfigValue // video.OPConfig
}

// InitConfig ...
func InitConfig(ctx context.Context, conf *Config) error {

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
		conf.Host = config.NewMultiStaticConfigValue(
			config.NewStaticConfigValue(
				func() interface{} {
					if len(conf.FileConfig.Host) == 0 {
						return nil
					}
					return conf.FileConfig.Host
				}),
			config.NewStaticConfigValue(
				func() interface{} { return os.Getenv(ENV_PORT_HTTP) }),
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
			))
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
		)
	}
	{
		conf.OPs = config.NewMultiStaticConfigValue(
			config.NewStaticConfigValue(
				func() interface{} {
					if conf.FileConfig.OPs == nil {
						return nil
					}
					return *conf.FileConfig.OPs
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
