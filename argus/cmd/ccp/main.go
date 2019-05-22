package main

import (
	"context"
	"net"
	"net/http"
	"runtime"

	"github.com/prometheus/client_golang/prometheus/promhttp"

	cconf "qbox.us/cc/config"
	jsonlog "qbox.us/http/audit/jsonlog.v3"

	"github.com/qiniu/db/mgoutil.v3"
	"github.com/qiniu/errors"
	"github.com/qiniu/http/httputil.v1"
	"github.com/qiniu/http/restrpc.v1"
	"github.com/qiniu/http/servestk.v1"
	"github.com/qiniu/log.v1"
	"github.com/qiniu/xlog.v1"

	"qiniu.com/argus/atserving/config"
	"qiniu.com/argus/ccp/manager"
	"qiniu.com/argus/ccp/manager/client"
	"qiniu.com/argus/ccp/manager/proto"
	"qiniu.com/argus/ccp/manager/proto/kodo"
	"qiniu.com/argus/ccp/manager/service"
	"qiniu.com/argus/com/pprof"
)

func main() {

	runtime.GOMAXPROCS(runtime.NumCPU())

	var (
		xl  = xlog.NewWith("main")
		ctx = xlog.NewContext(context.Background(), xl)
	)
	cconf.Init("f", "ccp", "ccp.conf")

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

	// FIXME: for Local DEBUG
	// conf.FileConfig.AuditLog.LogFile = "./CCP"

	al, logf, err := jsonlog.Open("CCP", &conf.FileConfig.AuditLog, nil)
	if err != nil {
		log.Fatal("jsonlog.Open failed:", errors.Detail(err))
	}
	defer logf.Close()

	mux := servestk.New(restrpc.NewServeMux(), al.Handler)
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("ok ccp"))
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

	xl.Info("Get EtcdConfig...")

	var (
		env EnvConfig
	)
	{
		vv, _ := conf.Env.Value(ctx)
		env = vv.(EnvConfig)
	}

	xl.Info("Got EtcdConfig")

	var (
		kodoSrcDAO proto.SrcDAO
		ruleDAO    proto.RuleDAO

		innerConfig    client.InnerConfig
		pfopRules      client.PfopRules
		bjobs          client.Bjobs
		reviewClient   client.ReviewClient
		notifyCallback client.NotifyCallback
		saveBack       client.SaveBack
		manualClient   client.ManualJobs
	)

	var (
		colls struct {
			KodoSrc mgoutil.Collection `coll:"kodosrc"`
			Rules   mgoutil.Collection `coll:"rules"`
		}
	)

	xl.Info("Init mongo...")

	{
		sess, err := mgoutil.Open(&colls, env.Mgo)
		if err != nil {
			log.Fatal("open mongo failed:", errors.Detail(err))
		}
		sess.SetPoolLimit(100)
		defer sess.Close()

		kodoSrcDAO = kodo.NewKodoSrcDAO(&colls.KodoSrc)
		srcDAOMap := make(map[string]proto.SrcDAO)
		srcDAOMap[proto.SRC_KODO] = kodoSrcDAO
		ruleDAO = proto.NewRuleDAO(&colls.Rules, srcDAOMap)
	}
	{
		innerConfig = client.NewInnerConfig(env.Saver, env.Manager.Host)
		pfopRules = client.NewPfopRules(innerConfig, env.Qconf, env.Pfop.UCHost)
		bjobs = client.NewBJobs(innerConfig, env.BJobs.Host)
		reviewClient = client.NewReviewClient(innerConfig, env.Review.Host)
		notifyCallback = client.NewNotifyCallback()
		saveBack = client.NewSaveBack(innerConfig, env.DomainApiHost, env.Qconf, env.Kodo)
		manualClient = client.NewManualJobs(innerConfig, env.Manual.Host)
	}
	var (
		rulesMng = manager.NewRules(ruleDAO, pfopRules, bjobs, manualClient, reviewClient)
		ruleServ = service.NewRuleService(rulesMng)
		msgServ  = service.NewMsgService(rulesMng, innerConfig,
			manualClient, reviewClient, notifyCallback, saveBack)
	)

	xl.Info("Register Router...")

	{
		router := restrpc.Router{
			PatternPrefix: "v1",
			Mux:           mux,
		}
		router.Register(ruleServ)
		router.Register(msgServ)
	}

	if err = http.ListenAndServe(
		net.JoinHostPort("0.0.0.0",
			// "10066"), // FIXME: for Local DEBUG
			config.GetString(ctx, conf.HTTPPort, "80")),
		mux); err != nil {
		log.Fatal("http.ListenAndServe", err)
	}
}
