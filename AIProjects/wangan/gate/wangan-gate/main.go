package main

import (
	"net/http"
	"runtime"
	"strconv"

	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/qiniu/errors"
	"github.com/qiniu/http/restrpc.v1"
	servestk "github.com/qiniu/http/servestk.v1"
	"github.com/qiniu/log.v1"
	cconf "qbox.us/cc/config"
	jsonlog "qbox.us/http/audit/jsonlog.v3"
	"qiniu.com/argus/AIProjects/wangan/gate"
)

type Config struct {
	HTTPPort   int             `json:"http_port"`
	Auditlog   jsonlog.Config  `json:"audit_log"`
	DebugLevel int             `json:"debug_level"`
	Gate       gate.GateConfig `json:"gate"`
}

func main() {
	runtime.GOMAXPROCS(runtime.NumCPU())

	var (
		conf Config
		mux  *servestk.ServeStack
	)

	cconf.Init("f", "wangan-gate", "wangan-gate")
	if err := cconf.Load(&conf); err != nil {
		log.Fatalf("Failed to load configure file, error: %v", err)
	}

	log.SetOutputLevel(conf.DebugLevel)
	log.Debugf("load conf %v", conf)

	al, logf, err := jsonlog.Open("WANGAN-GATE", &conf.Auditlog, nil)
	if err != nil {
		log.Fatal("jsonlog Open failed:", errors.Detail(err))
	}
	defer logf.Close()

	g := gate.NewGate(conf.Gate)
	server := gate.NewServer(g)

	mux = servestk.New(restrpc.NewServeMux(), func(w http.ResponseWriter, req *http.Request, f func(http.ResponseWriter, *http.Request)) {
		req.Header.Set("Authorization", "QiniuStub uid=1&ut=0")
		f(w, req)
	}, al.Handler)
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("ok " + "wanga-gate"))
	})
	mux.Handle("GET /metrics", promhttp.Handler())
	router := &restrpc.Router{
		PatternPrefix: "v1",
		Factory:       restrpc.Factory,
		Mux:           mux,
	}
	router.Register(server)
	if conf.HTTPPort <= 0 {
		conf.HTTPPort = 80
	}
	if err := http.ListenAndServe(
		"0.0.0.0:"+strconv.Itoa(conf.HTTPPort), mux,
	); err != nil {
		log.Fatal("http.ListenAndServe", err)
	}
}
