package main

import (
	"context"
	"net/http"
	"runtime"
	"strings"
	"time"

	"github.com/prometheus/client_golang/prometheus/promhttp"

	"github.com/qiniu/errors"
	"github.com/qiniu/http/restrpc.v1"
	"github.com/qiniu/http/servestk.v1"
	"github.com/qiniu/log.v1"
	xlog "github.com/qiniu/xlog.v1"

	"qbox.us/cc/config"

	jsonlog "qbox.us/http/audit/jsonlog.v3"
	"qiniu.com/argus/censor"
	_ "qiniu.com/argus/utility/censor"
)

func main() {

	runtime.GOMAXPROCS(runtime.NumCPU())
	var (
		xl  = xlog.NewWith("main")
		ctx = xlog.NewContext(context.Background(), xl)
	)

	config.Init("f", "argus-censor", "argus-censor.conf")
	var conf Config
	if err := config.Load(&conf); err != nil {
		log.Fatal("Failed to load configure file!")
	}
	if err := InitConfig(ctx, &conf); err != nil {
		xl.Fatalf("init config failed. %v", err)
	}
	if strings.TrimSpace(PORT_HTTP) != "" {
		conf.HTTPHost = "0.0.0.0:" + PORT_HTTP
	}

	log.SetOutputLevel(conf.DebugLevel)
	log.Debugf("load conf %#v", conf)

	var service censor.Service
	{
		var cfg ServiceConfig
		if vv, err := conf.Service.Value(ctx); err != nil {
			xl.Fatalf("load service config failed. %v", err)
		} else {
			cfg = vv.(ServiceConfig)
		}
		service.NewImageCensorClient = censor.NewImageCensorHTTPClient(
			cfg.ArgusImage.Host, time.Second*cfg.ArgusImage.TimeoutSecond)
		service.NewAsyncVideoClient = censor.NewAsyncVideoHTTPClient(
			cfg.ArgusVideo.Host, time.Second*cfg.ArgusVideo.TimeoutSecond)
		service.VideoConfig = cfg.Video
	}

	al, logf, err := jsonlog.Open("ARGUS-CENSOR", &conf.AuditLog, nil)
	if err != nil {
		log.Fatal("jsonlog.Open failed:", errors.Detail(err))
	}
	defer logf.Close()

	alMux := servestk.New(restrpc.NewServeMux(), al.Handler)
	alMux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("ok argus-censor"))
	})
	alMux.Handle("GET /metrics", promhttp.Handler())
	mux := http.NewServeMux()
	alMux.SetDefault(mux)

	router := restrpc.Router{
		PatternPrefix: "v1/censor",
		Mux:           alMux,
	}
	router.Register(service)

	if err := http.ListenAndServe(conf.HTTPHost, alMux); err != nil {
		log.Errorf("argus start error: %v", err)
	}
}
