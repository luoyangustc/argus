package main

import (
	"context"
	"net"
	"net/http"
	"os"
	"runtime"
	"strings"

	mgoutil "github.com/qiniu/db/mgoutil.v3"
	"github.com/qiniu/http/restrpc.v1"
	"github.com/qiniu/http/servestk.v1"
	"github.com/qiniu/log.v1"
	xlog "github.com/qiniu/xlog.v1"
	cconf "qbox.us/cc/config"
	jsonlog "qbox.us/http/audit/jsonlog.v3"
	"qiniu.com/argus/atserving/config"
	"qiniu.com/argus/censor_private/auth"
	"qiniu.com/argus/censor_private/dao"
	"qiniu.com/argus/censor_private/job"
	"qiniu.com/argus/censor_private/proto"
	"qiniu.com/argus/censor_private/service"
	"qiniu.com/argus/censor_private/util"
)

var (
	PORT_HTTP string
)

func init() {
	PORT_HTTP = os.Getenv("PORT_HTTP")
}

type Config struct {
	HTTPPort           string                    `json:"http_port"`
	AuditLog           jsonlog.Config            `json:"audit_log"`
	DebugLevel         int                       `json:"debug_level"`
	Mgo                *mgoutil.Config           `json:"mgo"`
	CensorImageService *proto.OuterServiceConfig `json:"censor_image_service"`
	CensorVideoService *proto.OuterServiceConfig `json:"censor_video_service"`
	Scenes             []proto.Scene             `json:"scenes"`
	MimeTypes          []proto.MimeType          `json:"mime_types"`
	SessionTimeout     int                       `json:"session_timeout_in_minute"`
	WorkerSize         int                       `json:"worker_size"`
	FileSaver          util.MinioConfig          `json:"file_saver"`
}

func main() {
	runtime.GOMAXPROCS(runtime.NumCPU())

	var (
		xl  = xlog.NewWith("main")
		ctx = xlog.NewContext(context.Background(), xl)
	)

	cconf.Init("f", "censor_private", "censor_private.conf")
	var conf Config
	if err := cconf.Load(&conf); err != nil {
		xl.Fatal("Failed to load configure file")
	}
	xl.Infof("load conf %#v", conf)

	log.SetOutputLevel(conf.DebugLevel)

	if strings.TrimSpace(PORT_HTTP) != "" {
		conf.HTTPPort = PORT_HTTP
	}

	al, logf, err := jsonlog.Open("censor_private", &conf.AuditLog, nil)
	if err != nil {
		xl.Fatal("jsonlog.Open failed:", err)
	}
	defer logf.Close()

	session := auth.NewAuth(ctx, conf.SessionTimeout*60)

	alMux := servestk.New(restrpc.NewServeMux(), session.Handler, al.Handler)
	alMux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("ok censor_private"))
	})

	mux := http.NewServeMux()
	alMux.SetDefault(mux)

	var (
		worker     job.IWorker
		fileSaver  util.FileSaver
		dispatcher *job.Dispatcher
	)

	{
		// init db
		xl.Info("mongo conf", config.DumpJsonConfig(conf.Mgo))
		sess, err := dao.SetUp(conf.Mgo)
		if err != nil {
			log.Fatal("open mongo failed:", err)
		}
		defer sess.Close()
	}
	{
		// init worker
		worker = job.NewWorker(ctx, &job.WorkerConfig{
			WorkerSize:         conf.WorkerSize,
			ImageServiceConfig: conf.CensorImageService,
			VideoServiceConfig: conf.CensorVideoService,
		})
	}
	{
		// init file server
		fileSaver, err = util.NewMinioSaver(conf.FileSaver)
		if err != nil {
			log.Fatal("failed to init minio saver : ", err)
		}
	}
	{
		// init uri proxy
		proxy := service.NewURIProxy("v1")

		// init service
		dispatcher = job.NewDispatcher(worker)
		svc, err := service.NewService(&service.ServiceConfig{
			Scenes:    conf.Scenes,
			MimeTypes: conf.MimeTypes,
		}, session, dispatcher, fileSaver, proxy)
		if err != nil {
			xl.Fatal(err)
		}

		router := restrpc.Router{
			PatternPrefix: "v1",
			Mux:           alMux,
		}
		router.Register(svc)
		router.Register(proxy)
	}
	{
		// init admin
		_, err = dao.UserDao.Find("admin")
		if err != nil {
			err = dao.UserDao.Insert(&proto.User{
				Id:       "admin",
				Password: util.Sha1("admin"),
				Roles:    []proto.Role{proto.RoleAdmin},
			})
			if err != nil {
				log.Fatal("init admin failed:", err)
			}
		}
	}
	{
		// restore monitor
		sets, err := dao.SetDao.Query(&dao.SetFilter{Status: proto.SetStatusRunning})
		if err != nil {
			log.Fatal("restore monitor failed:", err)
		}
		for _, s := range sets {
			err = dispatcher.Run(ctx, s)
			if err != nil {
				log.Fatal("run monitor failed:", err)
			}
		}
	}

	if err = http.ListenAndServe(
		net.JoinHostPort("0.0.0.0", conf.HTTPPort),
		alMux); err != nil {
		xl.Fatal("censor_private server start error: %v", err)
	}
}
