package main

import (
	"context"
	"net/http"
	"runtime"
	"strings"

	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/qiniu/db/mgoutil.v3"

	"github.com/qiniu/errors"
	"github.com/qiniu/http/restrpc.v1"
	"github.com/qiniu/http/servestk.v1"
	"github.com/qiniu/log.v1"
	xlog "github.com/qiniu/xlog.v1"

	"qbox.us/cc/config"
	"qbox.us/dht"

	jsonlog "qbox.us/http/audit/jsonlog.v3"
	"qiniu.com/argus/atserving/model"
	STS "qiniu.com/argus/sts/client"
	argus "qiniu.com/argus/utility"
	_ "qiniu.com/argus/utility/censor"
	"qiniu.com/argus/utility/server"
)

func main() {

	runtime.GOMAXPROCS(runtime.NumCPU())
	var (
		xl  = xlog.NewWith("main")
		ctx = xlog.NewContext(context.Background(), xl)
	)

	config.Init("f", "argus-util", "argus-util.conf")
	var conf Config
	if err := config.Load(&conf); err != nil {
		log.Fatal("Failed to load configure file!")
	}
	if err := InitConfig(ctx, &conf); err != nil {
		xl.Fatalf("init config failed. %v", err)
	}
	if USE_MOCK {
		conf.ArgGate.UseMock = true
	}
	if strings.TrimSpace(PORT_HTTP) != "" {
		conf.HTTPHost = "0.0.0.0:" + PORT_HTTP
	}
	if strings.TrimSpace(SERVING_GATE_HOST) != "" {
		conf.ArgGate.ServingHost = SERVING_GATE_HOST
		conf.Server.EvalDefault.Host = SERVING_GATE_HOST
	}

	log.SetOutputLevel(conf.DebugLevel)
	log.Debugf("load conf %#v", conf)

	if conf.BjrunMgo != nil {
		vv, _berr := conf.BjrunMgo.Value(ctx)
		if _berr == nil {
			mgo := vv.(mgoutil.Config)
			conf.ArgGate.MongConf = &mgo
		}
	}

	srv, err := argus.New(conf.ArgGate)
	if err != nil {
		log.Fatal("creating service failed:", errors.Detail(err))
	}
	var ip server.IImageParse
	if RUN_MODE != _ModeStandalone {
		var hosts []model.ConfigSTSHost
		if vv, err := conf.STSHosts.Value(ctx); err != nil {
			xl.Fatalf("load sts hosts failed. %v", err)
		} else {
			hosts = vv.([]model.ConfigSTSHost)
		}
		sts := func() STS.Client {
			var nodes = make(dht.NodeInfos, 0, len(hosts))
			for _, host := range hosts {
				nodes = append(nodes, dht.NodeInfo{Host: host.Host, Key: []byte(host.Key)})
			}
			lb, update := STS.NewLB(nodes, func() string { return xlog.GenReqId() }, nil)
			conf.STSHosts.Register(
				func(v interface{}) error {
					if v == nil {
						return nil
					}
					xl.Infof("sts update %#v", v)
					hosts := v.([]model.ConfigSTSHost)
					var nodes = make(dht.NodeInfos, 0, len(hosts))
					for _, host := range hosts {
						nodes = append(nodes, dht.NodeInfo{Host: host.Host, Key: []byte(host.Key)})
					}
					update(context.Background(), nodes)
					return nil
				},
			)
			return lb
		}()
		ip = server.ImageParse{Client: sts}
	} else {
		ip = server.NewStandaloneImageParser()
	}

	// --- TODO
	facegConf := argus.EvalsConfig{
		EvalConfig: argus.EvalConfig{
			Host: conf.ArgGate.ServingHost,
		},
		Evals: map[string]argus.EvalConfig{
			"facex-feature:v2": argus.EvalConfig{
				Host: conf.ArgGate.ServingHost,
			},
		},
	}
	manager, _ := argus.NewFaceGroupManagerInDB(conf.ArgGate.MongConf)
	facegsrv := argus.NewFaceGroupServic(facegConf, manager)
	// --- TODO

	{
		argus.RegisterFaceDetect()
		argus.RegisterFaceSim()
	}

	al, logf, err := jsonlog.Open("ARGUS-GATE", &conf.AuditLog, nil)
	if err != nil {
		log.Fatal("jsonlog.Open failed:", errors.Detail(err))
	}
	defer logf.Close()

	alMux := servestk.New(restrpc.NewServeMux(), al.Handler)
	alMux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("ok argus-gate"))
	})
	alMux.Handle("GET /metrics", promhttp.Handler())
	mux := http.NewServeMux()
	alMux.SetDefault(mux)

	router := restrpc.Router{
		PatternPrefix: "v1",
		Mux:           alMux,
	}
	router.Register(srv)
	router.Register(facegsrv) // TODO

	{
		s := server.DefaultServer.Init(conf.Server, ip)
		for _, handler := range s.Handlers() {
			router.Register(handler)
		}
	}

	if err := http.ListenAndServe(conf.HTTPHost, alMux); err != nil {
		log.Errorf("argus start error: %v", err)
	}
}
