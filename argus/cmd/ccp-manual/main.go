package main

import (
	"context"
	"encoding/json"
	"net/http"
	"runtime"

	"github.com/prometheus/client_golang/prometheus/promhttp"

	"github.com/qiniu/errors"
	filelog "github.com/qiniu/filelog/log"
	"github.com/qiniu/http/httputil.v1"
	"github.com/qiniu/http/restrpc.v1"
	"github.com/qiniu/http/servestk.v1"
	"github.com/qiniu/log.v1"
	"github.com/qiniu/xlog.v1"

	cconf "qbox.us/cc/config"
	jsonlog "qbox.us/http/audit/jsonlog.v3"

	"qiniu.com/argus/atserving/config"
	"qiniu.com/argus/ccp/manual"
	"qiniu.com/argus/ccp/manual/batch_entry_processor"
	"qiniu.com/argus/ccp/manual/client"
	"qiniu.com/argus/ccp/manual/dao"
	"qiniu.com/argus/ccp/manual/saver"
	"qiniu.com/argus/serving_eval"
)

func main() {

	runtime.GOMAXPROCS(runtime.NumCPU())

	var (
		xl   = xlog.NewWith("main")
		ctx  = xlog.NewContext(context.Background(), xl)
		conf Config
	)

	// 加载配置
	{
		cconf.Init("f", "ccp-manual", "ccp-manual.conf")
		if err := cconf.Load(&conf.FileConfig); err != nil {
			xl.Fatal("Failed to load configure file!")
		}
	}

	log.SetOutputLevel(conf.DebugLevel)
	if err := InitConfig(ctx, &conf); err != nil {
		xl.Fatalf("init config failed. %v", err)
	}

	xl.Info("loaded conf", dumps(conf))

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

	var (
		setDao         dao.ISetDAO
		streamEntryDao dao.IEntryDAO
		batchEntryDao  dao.IBatchEntryDAO
		err            error
	)
	{
		xl.Infof("begin init daos")
		setDao, err = dao.NewSetInMgo(*env.Mgo)
		if err != nil {
			xl.Infof("dao.NewSetInMgo error: %#v", err.Error())
			return
		}

		streamEntryDao, err = dao.NewEntryInMgo(*env.Mgo)
		if err != nil {
			xl.Infof("dao.NewEntryInMgo error: %#v", err.Error())
			return
		}

		batchEntryDao, err = dao.NewBatchEntryInMgo(*env.Mgo)
		if err != nil {
			xl.Infof("dao.NewBatchEntryInMgo error: %#v", err.Error())
			return
		}
		xl.Infof("dao finished")
	}

	var (
		handler   cap.IManualHandler
		capClient client.ICAPClient
	)
	{
		xl.Infof("client begin")
		capClient = client.NewCAPClient(env.CapConf)
		handler = cap.NewMaunalHandler(ctx, setDao, streamEntryDao, batchEntryDao, capClient)
	}

	var (
		bucketSaver = saver.NewBucketSaver(env.DomainApiHost, env.Qconf, env.Kodo, &setDao)
		service     = cap.NewService(ctx, handler)

		notifyclient = client.NewCapNotify(ctx, env.BatchEntryProcessorConf, env.CapConf, &setDao, &batchEntryDao, &bucketSaver)
	)

	//init batch entry job processor
	{
		batchEntryCron := batch_entry_processor.NewBatchEntryJobProcessor(
			env.BatchEntryProcessorConf,
			saver.NewKodoClient(env.DomainApiHost, env.Qconf, env.Kodo),
			&setDao, &batchEntryDao, &capClient, &bucketSaver,
		)

		batchEntryCron.Start(ctx)
		defer batchEntryCron.Close()
	}
	// init batch entry job result processor
	{
		batchEntryResultCron := batch_entry_processor.NewBatchEntryJobResult(*env.BatchEntryResultConf,
			&setDao, &batchEntryDao, &capClient, &notifyclient)

		batchEntryResultCron.Start(ctx)
		defer batchEntryResultCron.Close()
	}
	var mux *servestk.ServeStack
	{
		// 审计日志
		al, logf, err := jsonlog.Open("ARGUS_VFRAME", &conf.AuditLog, nil)
		if err != nil {
			log.Fatal("jsonlog.Open failed:", errors.Detail(err))
		}
		defer logf.Close()

		// run Service
		mux = servestk.New(restrpc.NewServeMux(), al.Handler)
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
		{
			router := &restrpc.Router{
				PatternPrefix: "/v1/ccp/manual",
				Factory:       restrpc.Factory,
				Mux:           mux,
			}
			router.Register(service)
		}

		// {
		// 	router := &restrpc.Router{
		// 		PatternPrefix: "/v1",
		// 		Factory:       restrpc.Factory,
		// 		Mux:           mux,
		// 	}
		// 	router.Register(notifyService)
		// }
	}

	if err := http.ListenAndServe(
		"0.0.0.0:"+config.GetString(ctx, conf.HTTPPort, "80"), mux,
	); err != nil {
		log.Fatal("http.ListenAndServe", err)
	}

}

func dumps(v interface{}) string {
	buf, _ := json.Marshal(v)
	return string(buf)
}

//----------------------------------------------------------------------------//
type fileLog struct {
	*filelog.Logger
}

func (l fileLog) Output(calldepth int, s string) error {
	return l.Log([]byte(s))
}
