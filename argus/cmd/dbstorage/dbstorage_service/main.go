package main

import (
	"context"
	"net"
	"net/http"
	"runtime"
	"runtime/debug"

	"qbox.us/errors"
	jsonlog "qbox.us/http/audit/jsonlog.v3"

	mgoutil "github.com/qiniu/db/mgoutil.v3"
	"github.com/qiniu/http/restrpc.v1"
	"github.com/qiniu/http/servestk.v1"
	"github.com/qiniu/log.v1"
	xlog "github.com/qiniu/xlog.v1"
	cconf "qbox.us/cc/config"
	"qiniu.com/argus/atserving/config"

	"qiniu.com/argus/dbstorage/dao"
	"qiniu.com/argus/dbstorage/proto"
	"qiniu.com/argus/dbstorage/service"
)

func main() {
	runtime.GOMAXPROCS(runtime.NumCPU())
	var (
		xl  = xlog.NewWith("main")
		ctx = xlog.NewContext(context.Background(), xl)
	)

	cconf.Init("f", "dbstorage_service", "dbstorage_service.conf")
	var conf Config
	if err := cconf.Load(&conf.FileConfig); err != nil {
		xl.Fatal("Failed to load configure file")
	}
	xl.Infof("load conf %#v", conf)

	if err := InitConfig(ctx, &conf); err != nil {
		xl.Fatalf("init config failed. %v", err)
	}
	log.SetOutputLevel(conf.DebugLevel)

	al, logf, err := jsonlog.Open("dbstorage", &conf.AuditLog, nil)
	if err != nil {
		log.Fatal("jsonlog.Open failed:", errors.Detail(err))
	}
	defer logf.Close()

	fns := []func(http.ResponseWriter, *http.Request, func(w http.ResponseWriter, req *http.Request)){}
	//if for feature_group_private, set/reset header
	if conf.IsPrivate {
		fns = append(fns, func(
			w http.ResponseWriter, req *http.Request, f func(http.ResponseWriter, *http.Request)) {
			req.Header.Set("Authorization", "QiniuStub uid=1&ut=0")
			f(w, req)
		})
	}
	fns = append(fns, al.Handler)

	alMux := servestk.New(restrpc.NewServeMux(), fns...)

	alMux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("ok dbstorage"))
	})

	mux := http.NewServeMux()
	alMux.SetDefault(mux)

	vv, _ := conf.Mgo.Value(ctx)
	mgo := vv.(mgoutil.Config)
	xl.Info("mongo conf", config.DumpJsonConfig(mgo))
	dao, err := dao.NewDbDao(&mgo)
	if nil != err {
		log.Error("NewDbDao failed", err)
		return
	}

	runningStatus := []proto.TaskStatus{proto.RUNNING, proto.PENDING, proto.STOPPING}
	if err = dao.ResetTask(ctx, proto.STOPPED, runningStatus...); err != nil {
		log.Error("ResetTask failed", err)
		return
	}

	vv, _ = conf.FeatureGroupService.Value(ctx)
	fg := vv.(proto.BaseServiceConfig)
	xl.Info("feature group service conf", config.DumpJsonConfig(fg))

	vv, _ = conf.ServingService.Value(ctx)
	serving := vv.(proto.BaseServiceConfig)
	xl.Info("serving service conf", config.DumpJsonConfig(serving))

	vv, _ = conf.ThreadNum.Value(ctx)
	thredNum := vv.(int)
	xl.Info("serving service conf", config.DumpJsonConfig(thredNum))

	vv, _ = conf.MaxParallelTaskNum.Value(ctx)
	parallel := vv.(int)
	xl.Info("serving service conf", config.DumpJsonConfig(parallel))

	taskService := service.NewTaskService(
		dao,
		&proto.TaskServiceConfig{
			FeatureGroupService: fg,
			ServingService:      serving,
			ThreadNum:           thredNum,
			MaxParallelTaskNum:  parallel,
			IsPrivate:           conf.IsPrivate,
		})

	router := restrpc.Router{
		PatternPrefix: "v1/face",
		Mux:           alMux,
	}
	router.Register(taskService)

	runtime.GC()
	debug.FreeOSMemory()

	if err = http.ListenAndServe(
		net.JoinHostPort("0.0.0.0",
			config.GetString(ctx, conf.HTTPPort, "80")),
		alMux); err != nil {
		log.Errorf("dbstorage server start error: %v", err)
	}
}
