package main

import (
	"net/http"
	"os"
	"runtime"
	"strings"

	"github.com/qiniu/errors"
	"github.com/qiniu/http/restrpc.v1"
	"github.com/qiniu/http/servestk.v1"
	"github.com/qiniu/log.v1"
	"qbox.us/cc/config"
	jsonlog "qbox.us/http/audit/jsonlog.v3"
	"qiniu.com/argus/AIProjects/chengyun/service"
)

var (
	PORT_HTTP string
)

func init() {
	PORT_HTTP = os.Getenv("PORT_HTTP")
}

type Config struct {
	AuditLog   jsonlog.Config    `json:"audit_log"`
	DebugLevel int               `json:"debug_level"`
	HTTPHost   string            `json:"http_host"`
	Server     service.Config    `json:"server"`
	Manager    service.MgrConfig `json:"manager"`
}

func main() {
	config.Init("f", "chengyun", "chengyun.conf")
	var (
		conf Config
		err  error
	)
	if err = config.Load(&conf); err != nil {
		log.Fatal("Failed to load configure file")
	}
	log.SetOutputLevel(conf.DebugLevel)
	log.Debugf("load conf %#v", conf)
	runtime.GOMAXPROCS(runtime.NumCPU())

	if strings.TrimSpace(PORT_HTTP) != "" {
		conf.HTTPHost = "0.0.0.0:" + PORT_HTTP
	}

	mgr, err := service.NewManager(conf.Manager)
	if err != nil {
		log.Fatal("service.NewManager:", errors.Detail(err))
	}

	srv, err := service.New(conf.Server, mgr)
	if err != nil {
		log.Fatal("server.New:", errors.Detail(err))
	}

	al, logf, err := jsonlog.Open("CHENGYUN", &conf.AuditLog, nil)
	if err != nil {
		log.Fatal("jsonlog.Open failed:", errors.Detail(err))
	}
	defer logf.Close()

	alMux := servestk.New(restrpc.NewServeMux(), al.Handler)
	alMux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("ok argus-chengyun"))
	})

	mux := http.NewServeMux()
	alMux.SetDefault(mux)

	router := restrpc.Router{
		PatternPrefix: "v1",
		Mux:           alMux,
	}
	router.Register(srv)
	if err := http.ListenAndServe(conf.HTTPHost, alMux); err != nil {
		log.Errorf("feature search server start error: %v", err)
	}

	log.Error("shutdown...")
}
