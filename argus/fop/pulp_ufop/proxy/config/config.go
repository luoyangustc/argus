package proxy_config

import (
	"encoding/json"
	"io/ioutil"
)

type Bucket struct {
	Ak     string `json:"ak"`
	Sk     string `json:"sk"`
	Name   string `json:"name"`
	Domain string `json:"domain"`
}

type Cmd struct {
	Name string `json:"name"`
	Url  string `json:"url"`
}

type Auth struct {
	Ak string `json:"ak"`
	Sk string `json:"sk"`
}

type Config struct {
	Bucket        Bucket `json:"bucket,omitEmpty"`
	Cmds          []Cmd  `json:"cmds,omitEmpty"`
	Port          int    `json:"port"`
	MaxConcurrent uint32 `json:"max_concurrent"`
	Auth          Auth   `json:"auth,omitEmpty"`
}

var globalConfig Config

func MaxConcurrent() uint32 {
	return globalConfig.MaxConcurrent
}

func LoadFromFile(filepath string) (*Config, error) {
	data, err := ioutil.ReadFile(filepath)
	if err != nil {
		return nil, err
	}
	return LoadFromData(data)
}

func LoadFromData(data []byte) (*Config, error) {
	return &globalConfig, json.Unmarshal(data, &globalConfig)
}
