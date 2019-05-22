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

	"qiniu.com/argus/argus/monitor"
	inspect "qiniu.com/argus/fop/bucket_inspect"
)

type Config struct {
	HTTPHost      string         `json:"http_host"`
	AuditLog      jsonlog.Config `json:"audit_log"`
	DebugLevel    int            `json:"debug_level"`
	InspectConfig inspect.Config `json:"inspect_config"`
}

var (
	PORT_HTTP string

	KODORS_HOST string
)

func init() {
	PORT_HTTP = os.Getenv("PORT_HTTP")
	KODORS_HOST = os.Getenv("KODORS_HOST")
}

func main() {

	runtime.GOMAXPROCS(runtime.NumCPU())

	config.Init("f", "bucket-inspect", "bucket-inspect.conf")
	var conf Config
	if err := config.Load(&conf); err != nil {
		log.Fatal("Failed to load configure file!")
	}
	if strings.TrimSpace(PORT_HTTP) != "" {
		conf.HTTPHost = "0.0.0.0:" + PORT_HTTP
	}

	log.SetOutputLevel(conf.DebugLevel)
	log.Debugf("load conf %#v", conf)

	// if strings.TrimSpace(KODORS_HOST) != "" {
	// 	conf.ProxyConfig.KodoRsHost = strings.TrimSpace(KODORS_HOST)
	// }

	service := inspect.NewService(conf.InspectConfig)

	al, logf, err := jsonlog.Open("BUCKET-INSPECT", &conf.AuditLog, nil)
	if err != nil {
		log.Fatal("jsonlog.Open failed:", errors.Detail(err))
	}
	defer logf.Close()

	alMux := servestk.New(restrpc.NewServeMux(), al.Handler)
	alMux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("ok bucket-inspect"))
	})
	alMux.Handle("GET /metrics", monitor.Handler())

	router := restrpc.Router{
		PatternPrefix: "",
		Mux:           alMux,
	}
	if err := router.ListenAndServe(conf.HTTPHost, service); err != nil {
		log.Errorf("argus start error: %v", err)
	}
}
