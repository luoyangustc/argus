package simple_service

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"runtime"
	"strings"
	"time"

	etcd "github.com/coreos/etcd/clientv3"
	"github.com/prometheus/client_golang/prometheus/promhttp"

	"github.com/qiniu/errors"
	"github.com/qiniu/http/restrpc.v1"
	"github.com/qiniu/http/servestk.v1"
	"github.com/qiniu/log.v1"
	xlog "github.com/qiniu/xlog.v1"
	cconf "qbox.us/cc/config"
	jsonlog "qbox.us/http/audit/jsonlog.v3"

	"qiniu.com/argus/atserving/config"
)

const (
	ENV_CONFIG_HOSTS = "CONFIG_HOSTS"
	ENV_PORT_HTTP    = "PORT_HTTP"
)

var (
	CONFIG_HOSTS []string
	PORT_HTTP    string
)

func init() {
	{
		hosts := os.Getenv("CONFIG_HOSTS")
		CONFIG_HOSTS = strings.Split(hosts, ",")
	}
	PORT_HTTP = os.Getenv("PORT_HTTP")
}

type Config struct {
	HTTPHost   string         `json:"http_host"`
	AuditLog   jsonlog.Config `json:"audit_log"`
	DebugLevel int            `json:"debug_level"`
}

func Main(app, patternPrefix string, service Service) {

	runtime.GOMAXPROCS(runtime.NumCPU())
	xl := xlog.NewWith("main")
	ctx := xlog.NewContext(context.Background(), xl)

	cconf.Init("f", app, fmt.Sprintf("%s.conf", app))
	var conf Config
	if err := cconf.Load(&conf); err != nil {
		xl.Fatal("Failed to load configure file!")
	}
	log.SetOutputLevel(conf.DebugLevel)
	log.Debugf("load conf %#v", conf)

	if strings.TrimSpace(PORT_HTTP) != "" {
		conf.HTTPHost = "0.0.0.0:" + PORT_HTTP
	}

	var etcdConfig etcd.Config
	etcdConfig.Endpoints = CONFIG_HOSTS
	etcdConfig.DialTimeout = time.Second
	client, err := etcd.New(etcdConfig)
	if err != nil {
		xl.Fatalf("new etcd client failed. %#v %v", etcdConfig, err)
	}

	{
		vv, err := config.NewStaticEtcdValue(
			etcd.NewKV(client), fmt.Sprintf("/ava/arugs/%s/config", app),
			func(bs []byte) (interface{}, error) {
				var v = service.Config()
				err := json.Unmarshal(bs, &v)
				return v, err
			},
		).Value(ctx)
		if err != nil {
			xl.Fatalf("get etcd config failed. %v", err)
		}
		if err = service.Init(vv); err != nil {
			log.Fatalf("init %s failed. %v", app, err)
		}
	}

	al, logf, err := jsonlog.Open(strings.ToUpper(app), &conf.AuditLog, nil)
	if err != nil {
		log.Fatal("jsonlog.Open failed:", errors.Detail(err))
	}
	defer logf.Close()

	alMux := servestk.New(restrpc.NewServeMux(), al.Handler)
	alMux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(fmt.Sprintf("ok %s", app)))
	})
	alMux.Handle("GET /metrics", promhttp.Handler())
	router := restrpc.Router{
		PatternPrefix: patternPrefix,
		Mux:           alMux,
	}

	if err := router.ListenAndServe(conf.HTTPHost, service); err != nil {
		log.Errorf("%s start error: %v", app, err)
	}
}
