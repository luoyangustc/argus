package main

import (
	"context"
	"net/http"
	"os"
	"runtime"
	"runtime/debug"
	"strings"

	"github.com/imdario/mergo"
	"github.com/qiniu/http/restrpc.v1"
	"github.com/qiniu/http/servestk.v1"
	"github.com/qiniu/log.v1"
	"qbox.us/cc/config"
	"qbox.us/errors"
	jsonlog "qbox.us/http/audit/jsonlog.v3"
	"qiniu.com/argus/feature_group_private/service"
)

var (
	PORT_HTTP string
)

func init() {
	PORT_HTTP = os.Getenv("PORT_HTTP")
}

type Config struct {
	AuditLog           jsonlog.Config            `json:"audit_log"`
	DebugLevel         int                       `json:"debug_level"`
	HTTPHost           string                    `json:"http_host"`
	BaseServiceConfig  service.BaseGroupsConfig  `json:"default_base_service_config"`
	ImageServiceConfig service.ImageGroupsConfig `json:"image_service_config"`
	FaceServiceConfig  service.FaceGroupsConfig  `json:"face_service_config"`
}

func main() {
	config.Init("f", "feature_group", "feature_group_private.conf")
	var conf Config
	if err := config.Load(&conf); err != nil {
		log.Fatal("Failed to load configure file")
	}
	mergo.Merge(&conf.ImageServiceConfig.BaseGroupsConfig, conf.BaseServiceConfig)
	mergo.Merge(&conf.FaceServiceConfig.BaseGroupsConfig, conf.BaseServiceConfig)

	log.SetOutputLevel(conf.DebugLevel)
	log.Debugf("load conf %#v", conf)
	runtime.GOMAXPROCS(runtime.NumCPU())
	ctx := context.Background()

	if strings.TrimSpace(PORT_HTTP) != "" {
		conf.HTTPHost = "0.0.0.0:" + PORT_HTTP
	}

	al, logf, err := jsonlog.Open("feature_group", &conf.AuditLog, nil)
	if err != nil {
		log.Fatal("jsonlog.Open failed:", errors.Detail(err))
	}
	defer logf.Close()

	alMux := servestk.New(restrpc.NewServeMux(), func(
		w http.ResponseWriter, req *http.Request, f func(http.ResponseWriter, *http.Request)) {
		req.Header.Set("Authorization", "QiniuStub uid=1&ut=0")
		f(w, req)
	}, al.Handler)

	alMux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("ok feature_group"))
	})

	mux := http.NewServeMux()
	alMux.SetDefault(mux)

	if conf.ImageServiceConfig.Enable {
		prefix := "v1/image"
		groups, err := service.NewBaseGroups(ctx, conf.ImageServiceConfig.BaseGroupsConfig, prefix)
		if nil != err {
			log.Error("NewBaseGroups failed", err)
			return
		}

		baseSrv, err := service.NewBaseService(ctx, groups, conf.ImageServiceConfig.BaseGroupsConfig)
		if err != nil {
			log.Fatal("NewBaseService failed:", errors.Detail(err))
		}
		imgSrv, err := service.NewImageService(ctx, groups, conf.ImageServiceConfig)
		if err != nil {
			log.Fatal("NewImageService failed:", errors.Detail(err))
		}
		router := restrpc.Router{
			PatternPrefix: prefix,
			Mux:           alMux,
		}
		router.Register(baseSrv)
		router.Register(imgSrv)
	}

	if conf.FaceServiceConfig.Enable {
		prefix := "v1/face"
		groups, err := service.NewBaseGroups(ctx, conf.FaceServiceConfig.BaseGroupsConfig, prefix)
		if nil != err {
			log.Error("NewBaseGroups failed", err)
			return
		}

		baseSrv, err := service.NewBaseService(ctx, groups, conf.FaceServiceConfig.BaseGroupsConfig)
		if err != nil {
			log.Fatal("NewBaseService failed:", errors.Detail(err))
		}
		faceSrv, err := service.NewFaceService(ctx, groups, conf.FaceServiceConfig)
		if err != nil {
			log.Fatal("NewFaceService failed:", errors.Detail(err))
		}
		router := restrpc.Router{
			PatternPrefix: prefix,
			Mux:           alMux,
		}
		router.Register(baseSrv)
		router.Register(faceSrv)
	}

	runtime.GC()
	debug.FreeOSMemory()

	if err := http.ListenAndServe(conf.HTTPHost, alMux); err != nil {
		log.Errorf("feature group private server start error: %v", err)
	}

	log.Error("shutdown...")
}
