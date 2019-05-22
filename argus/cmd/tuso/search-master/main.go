package main

import (
	"context"
	"net"
	"net/http"
	"runtime"
	"time"

	"qiniu.com/argus/tuso/client/image_feature"

	"qiniu.com/argus/tuso/client/tuso_hub"

	"github.com/prometheus/client_golang/prometheus/promhttp"

	cconf "qbox.us/cc/config"
	jsonlog "qbox.us/http/audit/jsonlog.v3"

	"github.com/qiniu/errors"
	"github.com/qiniu/http/httputil.v1"
	"github.com/qiniu/http/restrpc.v1"
	"github.com/qiniu/http/servestk.v1"
	"github.com/qiniu/log.v1"
	"github.com/qiniu/xlog.v1"

	"qiniu.com/argus/atserving/config"
	"qiniu.com/argus/atserving/simplequeue/sq"
	"qiniu.com/argus/bjob/job"
	"qiniu.com/argus/com/pprof"
	"qiniu.com/argus/tuso/tuso_job"
)

func main() {

	runtime.GOMAXPROCS(runtime.NumCPU())

	var (
		xl  = xlog.NewWith("main")
		ctx = xlog.NewContext(context.Background(), xl)
	)
	cconf.Init("f", "tuso-search-master", "tuso-search-master.conf")

	var conf = Config{}
	if err := cconf.Load(&conf.FileConfig); err != nil {
		xl.Fatal("Failed to load configure file!")
	}
	if err := InitConfig(ctx, &conf); err != nil {
		xl.Fatalf("init config failed. %v", err)
	}
	xl.Infof("load conf %#v", conf)
	if _, err := conf.HTTPPort.Value(ctx); err != nil {
		xl.Fatalf("need http port. %v", err)
	}

	al, logf, err := jsonlog.Open("TS_MASTER", &conf.FileConfig.AuditLog, nil)
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

	var mqConfig job.MgoAsMQConfig
	if vv, err := conf.MQ.Value(ctx); err != nil {
		xl.Fatalf("load sq hosts failed. %v", err)
	} else {
		mqConfig = vv.(job.MgoAsMQConfig)
	}

	var searchConfig tuso_job.SearchConfig
	if vv, err := conf.SearchConfig.Value(ctx); err != nil {
		xl.Fatalf("load search config failed. %v", err)
	} else {
		searchConfig = vv.(tuso_job.SearchConfig)
	}

	var featureApi image_feature.FeatureApiConfig
	if vv, err := conf.FeatureApi.Value(ctx); err != nil {
		xl.Fatal("load feature api config failed. %v", err)
	} else {
		featureApi = vv.(image_feature.FeatureApiConfig)
	}

	var internalApi tuso_hub.Config
	if vv, err := conf.InternalApi.Value(ctx); err != nil {
		xl.Fatal("load internal api config failed. %v", err)
	} else {
		internalApi = vv.(tuso_hub.Config)
	}

	xl.Info("load conf", config.DumpJsonConfig(searchConfig), config.DumpJsonConfig(internalApi), config.DumpJsonConfig(featureApi), config.DumpJsonConfig(mqConfig))

	var (
		node = func() job.MasterNodeConfig {
			var cmd = struct {
				Master job.MasterNodeConfig `json:"master"`
			}{}
			if vv, err := conf.Cmd.Value(ctx); err != nil {
				xl.Fatalf("load master hosts failed. %v", err)
			} else {
				cmd = vv.(struct {
					Master job.MasterNodeConfig `json:"master"`
				})
			}
			return cmd.Master
		}()
		mq, _  = job.NewMgoAsMQ(mqConfig, conf.FileConfig.Cmd)
		search = tuso_job.SearchNode{
			// put BatchConfig into config
			SearchConfig:    searchConfig,
			InternalApi:     tuso_hub.NewInternalHack(0, internalApi.Host, time.Second*time.Duration(internalApi.TimeoutSecond)),
			ImageFeatureApi: image_feature.NewFeatureApi(featureApi),
		}
		client = job.NewAsyncC(
			func() []*sq.Producer {
				var sqHosts []string
				if vv, err := conf.SQHosts.Value(ctx); err != nil {
					xl.Fatalf("load sq hosts failed. %v", err)
				} else {
					sqHosts = vv.([]string)
				}
				var ps = make([]*sq.Producer, 0, len(sqHosts))
				for _, host := range sqHosts {
					p, _ := sq.NewProducer(host) // TODO err
					ps = append(ps, p)
				}
				return ps
			}(),
			"first_argus_job_tuso_search",
			nil,
			func() string {
				if host, err := conf.LocalHost.Value(ctx); err != nil {
					xl.Fatalf("need callback host. %v", err)
				} else {
					return "http://" + host.(string)
				}
				return ""
			}(), "v1/search", mux,
		)
	)
	var master = job.NewMasterNode(node, mq, client, search)
	go master.Run()

	if err = http.ListenAndServe(
		net.JoinHostPort("0.0.0.0",
			config.GetString(ctx, conf.HTTPPort, "80")),
		mux); err != nil {
		log.Fatal("http.ListenAndServe", err)
	}
}
