package main

import (
	"context"
	"encoding/json"
	"net/http"
	_ "net/http/pprof"
	"os"
	"runtime"
	"strconv"

	"github.com/qiniu/errors"
	restrpc "github.com/qiniu/http/restrpc.v1"
	servestk "github.com/qiniu/http/servestk.v1"
	"github.com/qiniu/log.v1"
	xlog "github.com/qiniu/xlog.v1"
	cconf "qbox.us/cc/config"
	jsonlog "qbox.us/http/audit/jsonlog.v3"
	"qiniu.com/argus/AIProjects/mix"
	_ "qiniu.com/argus/licence"
	"qiniu.com/argus/serving_eval"
)

type Config struct {
	AuditLog   jsonlog.Config    `json:"audit_log"`
	DebugLevel int               `json:"debug_level"`
	Workspace  string            `json:"workspace"`
	Service    mix.ServiceConfig `json:"service"`
	HTTPPort   int               `json:"http_port"`
}

func main() {

	runtime.GOMAXPROCS(runtime.NumCPU())

	var (
		xl   = xlog.NewWith("main")
		ctx  = xlog.NewContext(context.Background(), xl)
		conf Config
	)
	_ = ctx

	// 加载配置
	{
		cconf.Init("f", "mix", "mix.conf")
		if err := cconf.Load(&conf); err != nil {
			xl.Fatal("Failed to load configure file!")
		}
	}

	var port string = os.Getenv("PORT_HTTP")
	if conf.HTTPPort > 0 {
		port = strconv.Itoa(conf.HTTPPort)
	}

	log.SetOutputLevel(conf.DebugLevel)
	xl.Info("loaded conf", dumps(conf))

	var mux *servestk.ServeStack
	{
		// 审计日志
		al, logf, err := jsonlog.Open("EVAL_"+eval.APP, &conf.AuditLog, nil)
		if err != nil {
			log.Fatal("jsonlog.Open failed:", errors.Detail(err))
		}
		defer logf.Close()

		// run Service
		mux = servestk.New(restrpc.NewServeMux(), al.Handler)
		mux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) {
			_, _ = w.Write([]byte("ok"))
		})

		router := &restrpc.Router{
			PatternPrefix: "v1",
			Factory:       restrpc.Factory,
			Mux:           mux,
		}
		service := mix.NewService(conf.Service)
		router.Register(service)
	}

	if err := http.ListenAndServe("0.0.0.0:"+port, mux); err != nil {
		log.Fatal("http.ListenAndServe", err)
	}
}

func dumps(v interface{}) string {
	buf, _ := json.Marshal(v)
	return string(buf)
}
