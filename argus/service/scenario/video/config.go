package video

import (
	"encoding/json"
	"os"

	qlog "github.com/qiniu/log.v1"
	"qbox.us/cc/config"

	"qiniu.com/argus/service/scenario/biz"
)

type Config struct {
	Name    string `json:"name"`
	Version string `json:"version"`

	Router     RouterConfig `json:"router"`
	Metrics    MetricConfig `json:"metrics"`
	DebugLevel int          `json:"debug_level"`
	Workspace  string       `json:"workspace"`

	Services map[string]*ServiceConfig `json:"services"` // map[ServingName]ServingConfig
}

type ServiceConfig struct {
	Name    string               `json:"name"`
	Version string               `json:"vesion"`
	Service json.RawMessage      `json:"service"`
	OPs     map[string]*OPConfig `json:"ops"`
}

type OPConfig struct {
	Name        string                            `json:"name"`
	Version     string                            `json:"vesion"`
	OP          json.RawMessage                   `json:"op"`
	EvalsDeploy biz.ServiceEvalDeployConfig       `json:"evals_deploy,omitempty"`
	Evals       map[string]*biz.ServiceEvalConfig `json:"evals"`
}

///////////////////////////////////////////////////////////////////////////////

func LoadConfig() Config {

	var conf Config
	config.Init("f", "app", "config.json")
	if err := config.Load(&conf); err != nil {
		qlog.Fatal("load config failed, err: ", err)
		os.Exit(2)
	}
	return conf
}
