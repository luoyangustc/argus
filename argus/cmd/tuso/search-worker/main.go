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

	"qiniu.com/argus/atserving/config"
	"qiniu.com/argus/atserving/model"
	"qiniu.com/argus/atserving/simplequeue/sq"
	"qiniu.com/argus/bjob/job"
	"qiniu.com/argus/com/pprof"
	"qiniu.com/argus/tuso/hub"
	"qiniu.com/argus/tuso/search"
	"qiniu.com/argus/tuso/tuso_job"
)

func main() {

	runtime.GOMAXPROCS(runtime.NumCPU())

	var (
		xl  = xlog.NewWith("main")
		ctx = xlog.NewContext(context.Background(), xl)
	)
	cconf.Init("f", "tuso-search-worker", "tuso-search-worker.conf")

	var conf = Config{}
	if err := cconf.Load(&conf.FileConfig); err != nil {
		xl.Fatal("Failed to load configure file!")
	}
	if err := InitConfig(ctx, &conf); err != nil {
		xl.Fatalf("init config failed. %v", err)
	}
	xl.Infof("load conf %#v", conf)

	al, logf, err := jsonlog.Open("JGATE", &conf.FileConfig.AuditLog, nil)
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

	var kodoConfig hub.KodoConfig
	if vv, err := conf.Kodo.Value(ctx); err != nil {
		xl.Fatalf("load Kodo config failed. %v", err)
	} else {
		kodoConfig = vv.(hub.KodoConfig)
		if len(kodoConfig.IoHost) == 0 {
			xl.Fatalf("Kodo io hosts can't be empty. %v", err)
		}
	}

	xl.Info("load conf", config.DumpJsonConfig(kodoConfig), config.DumpJsonConfig(conf))
	xl.Info("use distance algorithm", search.BestCosineDistance())

	{
		searchWorker := tuso_job.SearchWorker{
			SearchWorkerConfig: tuso_job.SearchWorkerConfig{},
			KodoConfig:         kodoConfig,
		}
		_, _ = job.NewAsyncS(
			func() []sq.ConsumerConfig {
				return []sq.ConsumerConfig{
					func() sq.ConsumerConfig {
						c1 := sq.NewConsumerConfig()

						var sqHosts []model.ConfigConsumer
						if vv, err := conf.SQHosts.Value(ctx); err != nil {
							xl.Fatalf("load sq hosts failed. %v", err)
						} else {
							sqHosts = vv.([]model.ConfigConsumer)
						}
						for _, v := range sqHosts {
							for _, d := range v.Addresses {
								c1.Addresses = append(c1.Addresses, d)
							}
						}

						//c1.MaxInFlight = &conf.SearchWorker.SQ.MaxInFlight
						c1.Topic = "first_argus_job_tuso_search"
						c1.Channel = c1.Topic + "_0"
						return c1
					}(),
				}
			}(),
			func(ctx context.Context, req []byte) ([]byte, error) {
				return searchWorker.Do(ctx, job.NewTask(req))
			},
		)
	}

	if err = http.ListenAndServe(
		net.JoinHostPort("0.0.0.0",
			config.GetString(ctx, conf.HTTPPort, "80")),
		mux); err != nil {
		log.Fatal("http.ListenAndServe", err)
	}
}