package main

import (
	"context"
	"flag"
	"log"

	"qbox.us/cc/config"
	"qiniu.com/argus/cmd/cli/serving-gate-auto/server"
)

func main() {
	config.Init("f", "serving-gate-auto", "serving-gate-auto.conf")
	flag.Parse()
	var cfg server.Config
	err := config.Load(&cfg)
	if err != nil {
		log.Panic(err)
	}
	s := server.New(cfg)
	s.Run(context.Background())
}
