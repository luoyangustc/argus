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
	filelog "github.com/qiniu/filelog/log"
	"github.com/qiniu/http/httputil.v1"
	"github.com/qiniu/http/restrpc.v1"
	"github.com/qiniu/http/servestk.v1"
	"github.com/qiniu/log.v1"
	"github.com/qiniu/xlog.v1"

	"qiniu.com/argus/atserving/config"
	"qiniu.com/argus/atserving/model"
	"qiniu.com/argus/atserving/simplequeue/sq"
	"qiniu.com/argus/bjob/job"
	"qiniu.com/argus/bjob/workers"
	"qiniu.com/argus/com/pprof"
)

func main() {

	runtime.GOMAXPROCS(runtime.NumCPU())

	var (
		xl  = xlog.NewWith("main")
		ctx = xlog.NewContext(context.Background(), xl)
	)
	cconf.Init("f", "inference-image-worker", "inference-image-worker.conf")

	var conf = Config{}
	if err := cconf.Load(&conf.FileConfig); err != nil {
		xl.Fatal("Failed to load configure file!")
	}
	if err := InitConfig(ctx, &conf); err != nil {
		xl.Fatalf("init config failed. %v", err)
	}
	xl.Infof("load conf %#v", conf)

	al, logf, err := jsonlog.Open("W", &conf.FileConfig.AuditLog, nil)
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

	var workerConfig job.WorkerNodeConfig
	if vv, err := conf.Worker.Value(ctx); err != nil {
		xl.Fatalf("load worker config failed. %v", err)
	} else {
		workerConfig = vv.(job.WorkerNodeConfig)
	}

	var iiConfig workers.InferenceImageConfig
	if vv, err := conf.Biz.Value(ctx); err != nil {
		xl.Fatalf("load biz config failed. %v", err)
	} else {
		iiConfig = vv.(workers.InferenceImageConfig)
	}

	xl.Info("load conf", config.DumpJsonConfig(iiConfig), config.DumpJsonConfig(conf))

	{
		lg, err := filelog.NewLogger(
			conf.SQLog.LogDir,
			conf.SQLog.LogPrefix,
			conf.SQLog.TimeMode,
			conf.SQLog.ChunkBits,
		)
		if err != nil {
			xl.Fatalf("filelog.Open failed. %#v %v", conf.SQLog, err)
		}
		defer lg.Close()
		sq.Logger = fileLog{lg}

		iiWorker := workers.NewInferenceImageWorker(iiConfig)
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
							c1.Addresses = append(c1.Addresses, v.Addresses...)
						}

						//c1.MaxInFlight = &conf.SearchWorker.SQ.MaxInFlight
						c1.Topic = "first_argus_jobw_" + JOB_WORKER
						c1.Channel = c1.Topic + "_0"

						if workerConfig.MaxInFlight > 0 {
							c1.MaxInFlight = &workerConfig.MaxInFlight
						} else {
							var maxInFlight = 100
							c1.MaxInFlight = &maxInFlight
						}
						return c1
					}(),
				}
			}(),
			func(ctx context.Context, req []byte) ([]byte, error) {
				return iiWorker.Do(ctx, job.NewTask(req))
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

type fileLog struct {
	*filelog.Logger
}

func (l fileLog) Output(calldepth int, s string) error {
	return l.Log([]byte(s))
}
