package main

import (
	"context"
	"net"
	"net/http"
	"runtime"
	"time"

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
	"qiniu.com/argus/ccp/review"
	"qiniu.com/argus/ccp/review/concerns"
	"qiniu.com/argus/ccp/review/dao"
	"qiniu.com/argus/serving_eval"
)

func main() {
	runtime.GOMAXPROCS(runtime.NumCPU())

	var (
		xl  = xlog.NewWith("main")
		ctx = xlog.NewContext(context.Background(), xl)
	)
	cconf.Init("f", "ccp-review", "review.conf")

	var conf = Config{}
	if err := cconf.Load(&conf.FileConfig); err != nil {
		xl.Fatal("Failed to load configure file!")
	}
	xl.Infof("load conf %#v", conf)
	if err := InitConfig(ctx, &conf); err != nil {
		xl.Fatalf("init config failed. %v", err)
	}
	log.SetOutputLevel(conf.DebugLevel)

	al, logf, err := jsonlog.Open("CCP_REVIEW", &conf.AuditLog, nil)
	if err != nil {
		xl.Fatalf("jsonlog.Open failed: %v", errors.Detail(err))
	}
	defer logf.Close()

	mux := servestk.New(restrpc.NewServeMux(), al.Handler)
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("ok " + eval.APP))
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

	var (
		env EnvConfig
	)

	{
		vv, err := conf.Env.Value(ctx)
		if err != nil {
			xl.Fatalf("env config load failed: %v", errors.Detail(err))
		}
		env = vv.(EnvConfig)
	}

	{
		sess, err := dao.SetUp(env.Mgo)
		if err != nil {
			log.Fatal("open mongo failed:", errors.Detail(err))
		}
		defer sess.Close()
	}

	{
		concerns.EntryCounterCacher = concerns.NewEntryCounterCacher(xl, 30*time.Second, 1024)
		concerns.EntryCounterCacher.Start()
		defer concerns.EntryCounterCacher.Close()
	}

	var (
		rService = review.NewService()
	)

	// init batch entry job cron
	{
		batchEntryCron := concerns.NewBatchEntryJobProcessor(
			ctx,
			concerns.NewKodoClient(env.DomainApiHost, env.Qconf, env.Kodo),
			10,
		)

		batchEntryCron.Start()
		defer batchEntryCron.Close()
	}

	{
		router := restrpc.Router{
			PatternPrefix: "v1",
			Mux:           mux,
		}
		router.Register(rService)
	}

	if err = http.ListenAndServe(
		net.JoinHostPort("0.0.0.0",
			// "10066"), // FIXME: for Local DEBUG
			config.GetString(ctx, conf.HTTPPort, "80")),
		mux); err != nil {
		log.Fatal("http.ListenAndServe", err)
	}
}
