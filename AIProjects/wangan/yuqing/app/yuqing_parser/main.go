package main

import (
	"context"
	"net/http"
	"runtime"
	"strconv"

	cconf "qbox.us/cc/config"
	"qiniu.com/argus/AIProjects/wangan/yuqing/parser"
	log "qiniupkg.com/x/log.v7"
)

type Config struct {
	HTTPPort   int `json:"http_port"`
	DebugLevel int `json:"debug_level"`

	Parser parser.ParserConfig `json:"parser"`
}

func main() {
	runtime.GOMAXPROCS(runtime.NumCPU())
	var (
		conf Config
		ctx  = context.Background()
	)
	cconf.Init("f", "yuqing_parser", "yuqing_parser")
	if err := cconf.Load(&conf); err != nil {
		log.Fatalf("Failed to load configure file, error: %v", err)
	}
	log.SetOutputLevel(conf.DebugLevel)
	log.Infof("load conf %v", conf)

	parser, err := parser.NewParser(conf.Parser)
	if err != nil {
		log.Fatalf("failed to create parser, err: %s", err.Error())
	}

	go parser.Run(ctx)
	go parser.Fetch(ctx)

	http.HandleFunc("/result", parser.GetResult)
	http.HandleFunc("/proxy", parser.Proxy)
	http.ListenAndServe("0.0.0.0:"+strconv.Itoa(conf.HTTPPort), nil)
}
