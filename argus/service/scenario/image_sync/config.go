package image_sync

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

	Router  RouterConfig `json:"router"`
	Metrics MetricConfig `json:"metrics"`

	Services map[string]*ServiceConfig `json:"services"` // map[ServingName]ServingConfig
}

type ServiceConfig struct {
	Name        string                            `json:"name"`
	Version     string                            `json:"vesion"`
	Service     json.RawMessage                   `json:"service"`
	Router      ServiceRouterConfig               `json:"router"`
	EvalsDeploy biz.ServiceEvalDeployConfig       `json:"evals_deploy,omitempty"`
	Evals       map[string]*biz.ServiceEvalConfig `json:"evals"`
}

///////////////////////////////////////////////////////////////////////////////

func LoadConfig() Config {

	var conf Config
	config.Init("f", "app", "config.json")
	if err := config.Load(&conf); err != nil {
		qlog.Fatalf("load config failed, err: ", err)
		os.Exit(2)
	}
	return conf
}
