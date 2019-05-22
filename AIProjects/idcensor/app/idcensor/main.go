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
	"qiniu.com/argus/AIProjects/idcensor/service"
)

type Config struct {
	Version    string          `json:"version"`
	HTTPHost   string          `json:"http_host"`
	AuditLog   jsonlog.Config  `json:"audit_log"`
	DebugLevel int             `json:"debug_level"`
	Server     idcensor.Config `json:"server"`
}

var (
	PORT_HTTP                     string
	AIPROJECT_IDCENSOR_HANGHUI_AK string
	AIPROJECT_IDCENSOR_HANGHUI_SK string

	AIPROJECT_IDCENSOR_AK string
	AIPROJECT_IDCENSOR_SK string
)

func init() {
	PORT_HTTP = os.Getenv("PORT_HTTP")
	AIPROJECT_IDCENSOR_HANGHUI_AK = os.Getenv("AIPROJECT_IDCENSOR_HANGHUI_AK")
	AIPROJECT_IDCENSOR_HANGHUI_SK = os.Getenv("AIPROJECT_IDCENSOR_HANGHUI_SK")
	AIPROJECT_IDCENSOR_AK = os.Getenv("AIPROJECT_IDCENSOR_AK")
	AIPROJECT_IDCENSOR_SK = os.Getenv("AIPROJECT_IDCENSOR_SK")
}

func main() {

	runtime.GOMAXPROCS(runtime.NumCPU())

	config.Init("f", "idcensor", "idcensor.conf")
	var conf Config
	if err := config.Load(&conf); err != nil {
		log.Fatal("Failed to load configure file!")
	}

	if strings.TrimSpace(PORT_HTTP) != "" {
		conf.HTTPHost = "0.0.0.0:" + PORT_HTTP
	}
	if strings.TrimSpace(AIPROJECT_IDCENSOR_HANGHUI_AK) != "" {
		conf.Server.HanghuiAK = strings.TrimSpace(AIPROJECT_IDCENSOR_HANGHUI_AK)
	}
	if strings.TrimSpace(AIPROJECT_IDCENSOR_HANGHUI_SK) != "" {
		conf.Server.HanghuiSK = strings.TrimSpace(AIPROJECT_IDCENSOR_HANGHUI_SK)
	}
	if strings.TrimSpace(AIPROJECT_IDCENSOR_AK) != "" {
		conf.Server.AK = strings.TrimSpace(AIPROJECT_IDCENSOR_AK)
	}
	if strings.TrimSpace(AIPROJECT_IDCENSOR_SK) != "" {
		conf.Server.SK = strings.TrimSpace(AIPROJECT_IDCENSOR_SK)
	}

	log.SetOutputLevel(conf.DebugLevel)
	log.Debugf("load conf %#v", conf)

	srv := idcensor.NewServer(conf.Server)

	al, logf, err := jsonlog.Open("ARGUS-FOP", &conf.AuditLog, nil)
	if err != nil {
		log.Fatal("jsonlog.Open failed:", errors.Detail(err))
	}
	defer logf.Close()

	alMux := servestk.New(restrpc.NewServeMux(), al.Handler)
	alMux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("ok idcensor"))
	})

	router := restrpc.Router{
		PatternPrefix: conf.Version,
		Mux:           alMux,
	}

	if err := router.ListenAndServe(conf.HTTPHost, router.Register(srv)); err != nil {
		log.Errorf("idcensor start error: %v", err)
	}
}
