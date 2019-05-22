package main

import (
	"net/http"
	"os"
	"runtime"
	"strings"

	mgoutil "github.com/qiniu/db/mgoutil.v3"
	"github.com/qiniu/errors"
	"github.com/qiniu/http/restrpc.v1"
	"github.com/qiniu/http/servestk.v1"
	"github.com/qiniu/log.v1"

	config_loader "qbox.us/cc/config"
	jsonlog "qbox.us/http/audit/jsonlog.v3"
)

type Config struct {
	HTTPHost       string         `json:"http_host"`
	AuditLog       jsonlog.Config `json:"audit_log"`
	DebugLevel     int            `json:"debug_level"`
	FfmpegCmd      string         `json:"ffmpeg_cmd"`
	FfmpegArgsTpl  string         `json:"ffmpeg_args_tpl"`
	AIHost         string         `json:"ai_host"`
	DownstreamHost string         `json:"downstream_host"`
	ClusterArgs    struct {
		Size      int `json:"size"`
		Precision int `json:"precision"`
		Dimension int `json:"dimension"`
	} `json:"cluster_args"`
	MaxProcessNum int            `json:"max_process_num"`
	MgoConfig     mgoutil.Config `json:"mgo_config"`
}

var (
	PORT_HTTP string
)

func init() {
	PORT_HTTP = os.Getenv("PORT_HTTP")
}

func main() {

	runtime.GOMAXPROCS(runtime.NumCPU())

	// log config from file
	var conf Config
	config_loader.Init("f", "video-stream-backend", "video-stream-backend.conf")
	if err := config_loader.Load(&conf); err != nil {
		log.Fatal("Failed to load configure file!")
	}
	log.SetOutputLevel(conf.DebugLevel)
	log.Debugf("Load conf %#v", conf)

	// setup audit log
	al, logf, err := jsonlog.Open("video-stream-backend", &conf.AuditLog, nil)
	if err != nil {
		log.Fatal("jsonlog.Open failed:", errors.Detail(err))
	}
	defer logf.Close()

	if strings.TrimSpace(PORT_HTTP) != "" {
		conf.HTTPHost = "0.0.0.0:" + PORT_HTTP
	}

	var alMux *servestk.ServeStack
	// run Service
	alMux = servestk.New(restrpc.NewServeMux(), al.Handler)
	alMux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("ok video-stream-backend"))
	})

	srv, err := NewService(conf)
	if err != nil {
		log.Fatal("server init failed")
	}

	mux := http.NewServeMux()
	alMux.SetDefault(mux)

	router := restrpc.Router{
		PatternPrefix: "v1",
		Mux:           alMux,
	}
	router.Register(srv)
	if err := http.ListenAndServe(conf.HTTPHost, alMux); err != nil {
		log.Errorf("video-stream-backend server start error: %v", err)
	}
	log.Error("shutdown...")
}
