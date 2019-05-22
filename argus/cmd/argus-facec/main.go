package main

import (
	"net/http"
	"os"
	"runtime"
	"strings"
	"time"

	"github.com/qiniu/errors"
	"github.com/qiniu/http/restrpc.v1"
	"github.com/qiniu/http/servestk.v1"
	"github.com/qiniu/log.v1"
	"qbox.us/cc/config"
	jsonlog "qbox.us/http/audit/jsonlog.v3"

	"qiniu.com/argus/argus/facec"
	"qiniu.com/argus/argus/monitor"
)

var (
	USE_MOCK          bool
	PORT_HTTP         string
	SERVING_GATE_HOST string
	MONGO_HOST        string
)

func init() {
	PORT_HTTP = os.Getenv("PORT_HTTP")
	SERVING_GATE_HOST = os.Getenv("SERVING_GATE_HOST")
	MONGO_HOST = os.Getenv("MONGO_HOST")
}

type Config struct {
	Version    string         `json:"version"`
	HTTPHost   string         `json:"http_host"`
	AuditLog   jsonlog.Config `json:"audit_log"`
	Server     facec.Config   `json:"server"`
	DebugLevel int            `json:"debug_level"`
}

func main() {

	config.Init("f", "argus-gate", "argus-gate.conf")
	var conf Config
	if err := config.Load(&conf); err != nil {
		log.Fatal("Failed to load configure file!")
	}
	if os.Getenv("USE_MOCK") == "true" {
		conf.Server.UseMock = true
	}
	log.SetOutputLevel(conf.DebugLevel)
	log.Debugf("load conf %#v", conf)
	runtime.GOMAXPROCS(runtime.NumCPU())

	if strings.TrimSpace(PORT_HTTP) != "" {
		conf.HTTPHost = "0.0.0.0:" + PORT_HTTP
	}
	if strings.TrimSpace(SERVING_GATE_HOST) != "" {
		conf.Server.Hosts.FacexDet = SERVING_GATE_HOST
		conf.Server.Hosts.FacexFeature = SERVING_GATE_HOST
		conf.Server.Hosts.FacexCluster = SERVING_GATE_HOST
	}
	if strings.TrimSpace(MONGO_HOST) != "" {
		conf.Server.MgoConfig.Host = MONGO_HOST
	}

	srv, err := facec.New(conf.Server)
	if err != nil {
		log.Fatal("argus.New:", errors.Detail(err))
	}
	al, logf, err := jsonlog.Open("ARGUS-FACEC", &conf.AuditLog, nil)
	if err != nil {
		log.Fatal("jsonlog.Open failed:", errors.Detail(err))
	}
	defer logf.Close()

	alMux := servestk.New(restrpc.NewServeMux(), al.Handler)
	alMux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("ok argus-facex"))
	})
	alMux.Handle("GET /metrics", monitor.Handler())
	router := restrpc.Router{
		PatternPrefix: conf.Version,
		Mux:           alMux,
	}

	facec.NewFeatureWorker(
		&facec.FeatureWorkerConfig{
			URL:        url(conf.Server.Hosts.FacexFeature, "v1/eval/facex-feature"),
			Period:     time.Second,
			Concurrent: 3,
			BatchSize:  10,
		})
	facec.NewClusterWorker(
		&facec.ClusterWorkerConfig{
			URL:        url(conf.Server.Hosts.FacexCluster, "v1/eval/facex-cluster"),
			Period:     time.Second,
			Concurrent: 2,
		})

	if err := router.ListenAndServe(conf.HTTPHost, srv); err != nil {
		log.Errorf("argus-facec start error: %v", err)
	}
}

func url(prefix, path string) string {
	if strings.HasSuffix(prefix, "/") {
		return prefix + path
	}

	return prefix + "/" + path
}
