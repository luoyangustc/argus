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

	"qiniu.com/argus/argus/com/dht"
	"qiniu.com/argus/atserving/config"
	"qiniu.com/argus/com/pprof"
	FG "qiniu.com/argus/feature_group"
	FACEG "qiniu.com/argus/feature_group/faceg"
)

func main() {

	runtime.GOMAXPROCS(runtime.NumCPU())

	var (
		xl  = xlog.NewWith("main")
		ctx = xlog.NewContext(context.Background(), xl)
	)
	cconf.Init("f", "face_group_worker", "face_group_worker.conf")

	var conf = Config{}
	if err := cconf.Load(&conf.FileConfig); err != nil {
		xl.Fatal("Failed to load configure file!")
	}
	xl.Infof("load conf %#v", conf)
	if err := InitConfig(ctx, &conf); err != nil {
		xl.Fatalf("init config failed. %v", err)
	}
	log.SetOutputLevel(conf.DebugLevel)
	if _, err := conf.HTTPPort.Value(ctx); err != nil {
		xl.Fatalf("need http port. %v", err)
	}

	al, logf, err := jsonlog.Open("FG_WORKER", &conf.FileConfig.AuditLog, nil)
	if err != nil {
		log.Fatal("jsonlog.Open failed:", errors.Detail(err))
	}
	defer logf.Close()

	mux := servestk.New(restrpc.NewServeMux(), al.Handler)
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("ok face-group-worker"))
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

	var worker FG.Worker
	{
		vv, _ := conf.Storage.Value(ctx)
		sConfig := vv.(FG.StorageConfig)
		xl.Info("storage conf", config.DumpJsonConfig(sConfig))
		vv2, _ := conf.Fetch.Value(ctx)
		fConfig := vv2.(struct {
			URL string `json:"url"`
		})
		xl.Info("fetch conf", config.DumpJsonConfig(fConfig))

		fetch := FG.HubFeatureFetch{URL: fConfig.URL}
		worker.Memory = FG.NewMemory(sConfig,
			func(ctx context.Context, key FG.Key) ([]byte, error) {
				return fetch.Fetch(ctx, key)
			},
		)
	}

	{
		router := restrpc.Router{
			PatternPrefix: "v1",
			Mux:           mux,
		}
		router.Register(worker)
	}

	{
		// register local host to etcd
		var url string
		if host, err := conf.LocalHost.Value(ctx); err != nil {
			xl.Fatalf("need local host. %v", err)
		} else {
			url = "http://" + host.(string)
		}

		d := dht.NewETCD(client, FACEG.FACE_GROUP_WORKER_NODE_PREFIX)
		if err := d.Register(ctx, url, 10); err != nil { // 增加TTL，减少波动
			xl.Errorf("etcd register fail. %s %v", url, err)
		} else {
			xl.Infof("etcd register. %s", url)
		}
	}

	if err = http.ListenAndServe(
		net.JoinHostPort("0.0.0.0",
			config.GetString(ctx, conf.HTTPPort, "80")),
		mux); err != nil {
		log.Fatal("http.ListenAndServe", err)
	}
}
