package main

import (
	"flag"
	"os"
	"strconv"

	"time"

	"qiniu.com/argus/fop/pulp_ufop/proxy/client"
	"qiniu.com/argus/fop/pulp_ufop/proxy/config"
	"qiniu.com/argus/fop/pulp_ufop/proxy/resource"
	"qiniu.com/argus/fop/pulp_ufop/proxy/server"
)

var (
	configPath = flag.String("f", "", "the/path/to/configfile")
)

func main() {
	flag.Parse()
	if *configPath == "" {
		println("configure file is empty")
		return
	}

	config, err := proxy_config.LoadFromFile(*configPath)
	if err != nil {
		println("load configure error:" + err.Error())
		return
	}

	if config.MaxConcurrent <= 0 {
		println("max concurrent must be positive count")
		return
	}

	auth := &config.Auth

	if auth.Ak == "" || auth.Sk == "" {
		println("please initialize the auth (ak/sk)")
		return
	}

	r, err := proxy_resource.CreateBucketResource(&config.Bucket)
	if err != nil {
		println("create bucket resource error:", err)
		return
	}

	port := os.Getenv("PORT_HTTP")
	if port == "" {
		port = "9100"
	}

	portNum, err := strconv.Atoi(port)
	if err != nil {
		println("convert port to number error", err)
		return
	}
	proxy_server.Run(portNum,
		proxy_client.CreateAuthClient(auth.Ak, auth.Sk, 0, 0, 120*time.Second),
		r, config.Cmds, config.MaxConcurrent)
}
