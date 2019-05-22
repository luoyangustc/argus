package main

import (
	"context"
	"net"
	"net/http"
	"runtime"

	"github.com/prometheus/client_golang/prometheus/promhttp"

	cconf "qbox.us/cc/config"
	jsonlog "qbox.us/http/audit/jsonlog.v3"

	"github.com/qiniu/errors"
	"github.com/qiniu/http/httputil.v1"
	"github.com/qiniu/http/restrpc.v1"
	"github.com/qiniu/http/servestk.v1"
	"github.com/qiniu/log.v1"
	"github.com/qiniu/xlog.v1"

	"github.com/qiniu/db/mgoutil.v3"
	"qiniu.com/argus/atserving/config"
	"qiniu.com/argus/com/pprof"
	"qiniu.com/argus/tuso/hub"
)

func main() {

	runtime.GOMAXPROCS(runtime.NumCPU())

	var (
		xl  = xlog.NewWith("main")
		ctx = xlog.NewContext(context.Background(), xl)
	)
	cconf.Init("f", "tuso-hub-gate", "tuso-hub-gate.conf")

	var conf = Config{}
	if err := cconf.Load(&conf.FileConfig); err != nil {
		xl.Fatal("Failed to load configure file!")
	}
	xl.Infof("load conf %#v", conf)
	if err := InitConfig(ctx, &conf); err != nil {
		xl.Fatalf("init config failed. %v", err)
	}
	if _, err := conf.HTTPPort.Value(ctx); err != nil {
		xl.Fatalf("need http port. %v", err)
	}

	al, logf, err := jsonlog.Open("HUB-GATE", &conf.FileConfig.AuditLog, nil)
	if err != nil {
		log.Fatal("jsonlog.Open failed:", errors.Detail(err))
	}
	defer logf.Close()

	mux := servestk.New(restrpc.NewServeMux(), al.Handler)
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("ok serving-gate"))
	})
	mux.Handle("GET /metrics", promhttp.Handler())
	mux.HandleFunc("GET /configs", func(w http.ResponseWriter, r *http.Request) {
		m, err := config.DumpSimpleStruct(context.Background(), conf)
		if err != nil {
			httputil.ReplyErr(w, http.StatusInternalServerError, err.Error())
		} else {
			httputil.Reply(w, http.StatusOK, m)
		}
	})
	{
		mux2 := http.NewServeMux()
		mux.SetDefault(mux2)
		pprof.Handle(mux2, "")
	}

	hubCfg := hub.Config{}
	if vv, err := conf.Mgo.Value(ctx); err != nil {
		xl.Fatalf("load mgo hosts failed. %v", err)
	} else {
		hubCfg.Mgo = vv.(mgoutil.Config)
	}

	if vv, err := conf.Kodo.Value(ctx); err != nil {
		xl.Fatalf("load Kodo hosts failed. %v", err)
	} else {
		hubCfg.Kodo = vv.(hub.KodoConfig)
	}

	if vv, err := conf.JobGateApi.Value(ctx); err != nil {
		xl.Fatalf("load JobGateApi hosts failed. %v", err)
	} else {
		hubCfg.JobGateApi = vv.(hub.JobGateApiConfig)
	}

	xl.Info("load conf", config.DumpJsonConfig(hubCfg))

	server, internalServer, _, err := hub.New(hubCfg)

	if err != nil {
		log.Panicln(err)
	}

	{
		router := restrpc.Router{
			PatternPrefix: "v1",
			Mux:           mux,
		}
		router.Register(server)
	}

	{
		router := restrpc.Router{
			PatternPrefix: "hub_admin",
			Mux:           mux,
		}
		router.Register(internalServer)
	}

	if err = http.ListenAndServe(
		net.JoinHostPort("0.0.0.0",
			config.GetString(ctx, conf.HTTPPort, "80")),
		mux); err != nil {
		log.Fatal("http.ListenAndServe", err)
	}
}
