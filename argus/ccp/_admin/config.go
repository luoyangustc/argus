package main

import (
	mgoutil "github.com/qiniu/db/mgoutil.v3"
	cconf "qbox.us/cc/config"
	"qbox.us/qconf/qconfapi"
	"qiniupkg.com/api.v7/kodo"
)

type _AdminConfig struct {
	Qconf qconfapi.Config `json:"qconf"`
	Kodo  kodo.Config     `json:"kodo"`
	CcpM  struct {
		Mgo mgoutil.Config `json:"mgo"`
	} `json:"ccp_m"`
}

var (
	AdminConfig _AdminConfig
)

func (cfg *_AdminConfig) Load(confPath string) error {
	return cconf.LoadEx(cfg, confPath)
}
