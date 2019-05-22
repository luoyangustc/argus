package main

import (
	"context"
	"encoding/json"
	"net/http"
	"runtime"

	"github.com/prometheus/client_golang/prometheus/promhttp"

	"github.com/qiniu/db/mgoutil.v3"
	"github.com/qiniu/errors"
	"github.com/qiniu/http/httputil.v1"
	"github.com/qiniu/http/restrpc.v1"
	"github.com/qiniu/http/servestk.v1"
	"github.com/qiniu/log.v1"
	"github.com/qiniu/xlog.v1"

	cconf "qbox.us/cc/config"
	jsonlog "qbox.us/http/audit/jsonlog.v3"

	"qiniu.com/argus/atserving/config"
	"qiniu.com/argus/cap"
	"qiniu.com/argus/cap/auditor"
	"qiniu.com/argus/cap/dao"
	"qiniu.com/argus/cap/job"
	"qiniu.com/argus/cap/sand"
	"qiniu.com/argus/cap/task"
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
		cconf.Init("f", "cap", "cap.conf")
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

	xl.Infof("begin init daos")
	var (
		jobDao     dao.IJobDAO
		taskDao    dao.ITaskDAO
		auditorDao dao.IAuditorDAO
		groupDao   dao.IGroupDAO
		labelDao   dao.ILabelDAO
	)
	{

		mgoSessionPoolLimit := 100
		var (
			colls struct {
				Auditors mgoutil.Collection `coll:"auditors"`
				Labels   mgoutil.Collection `coll:"labels"`
				Groups   mgoutil.Collection `coll:"groups"`
			}
		)
		sess, err := mgoutil.Open(&colls, &env.Mgo.Mgo)
		if err != nil {
			xl.Infof("dao.NewJobDao error: %#v", err.Error())
			return
		}
		sess.SetPoolLimit(mgoSessionPoolLimit)
		defer sess.Close()

		jobDao, err = dao.NewJobDao(env.Mgo)
		if err != nil {
			xl.Infof("dao.NewJobDao error: %#v", err.Error())
			return
		}
		taskDao, err = dao.NewTaskDao(env.Mgo)
		if err != nil {
			xl.Infof("dao.NewTaskDao error: %#v", err.Error())
			return
		}

		auditorDao = dao.NewAuditorInMgo(&colls.Auditors)
		groupDao = dao.NewGroupInMgo(&colls.Groups)
		labelDao = dao.NewLabelInMgo(&colls.Labels)

		xl.Infof("dao finished")
	}

	var (
		auditorHandler auditor.IAuditor
		auditService   cap.IAuditService

		sandMixer   sand.ISandMixer
		sandService cap.ISandService

		jobHandler job.IJobHandler
		jobService cap.IJobService

		resultHandler task.IBatchResultHandler
		resultService cap.IResultService

		adminService cap.IAdminService
	)
	{
		sandMixer = sand.NewSandMixer(env.SandPattern)
		sandService, _ = cap.NewSandService(sandMixer)

		auditorHandler = auditor.NewAuditor(taskDao, auditorDao, groupDao, labelDao, sandMixer, env.Auditor)
		auditService, _ = cap.NewAuditService(auditorHandler)

		jobHandler = job.NewJobHandler(jobDao, taskDao)
		jobService = cap.NewJobService(&jobHandler)

		resultHandler = task.NewBatchResult(jobDao, taskDao)
		resultService = cap.NewResultService(resultHandler, jobDao, taskDao)

		adminService, _ = cap.NewAdminService(labelDao, groupDao, auditorDao)
	}
	// 初始化沙子库
	{
		for _, file := range env.SandFiles {
			log.Println("Loading sand file:", file)
			err := sandMixer.AddSandFileByURL(file)
			if err != nil {
				log.Warn(err)
			}
		}
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
				PatternPrefix: "/v1/cap",
				Factory:       restrpc.Factory,
				Mux:           mux,
			}
			router.Register(jobService)
		}
		{
			router := &restrpc.Router{
				PatternPrefix: "v1/cap",
				Factory:       restrpc.Factory,
				Mux:           mux,
			}
			router.Register(auditService)
		}
		{
			router := &restrpc.Router{
				PatternPrefix: "v1/cap",
				Factory:       restrpc.Factory,
				Mux:           mux,
			}
			router.Register(sandService)
		}
		{
			router := &restrpc.Router{
				PatternPrefix: "/v1/cap",
				Factory:       restrpc.Factory,
				Mux:           mux,
			}
			router.Register(resultService)
		}
		{
			router := &restrpc.Router{
				PatternPrefix: "v1/admin",
				Factory:       restrpc.Factory,
				Mux:           mux,
			}
			router.Register(adminService)
		}
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
