package main

import (
	"context"
	"net"
	"net/http"
	"reflect"
	"runtime"
	"strings"

	"github.com/prometheus/client_golang/prometheus/promhttp"

	"github.com/qiniu/errors"
	"github.com/qiniu/http/restrpc.v1"
	"github.com/qiniu/http/rpcutil.v1"
	"github.com/qiniu/http/servestk.v1"
	"github.com/qiniu/log.v1"
	xlog "github.com/qiniu/xlog.v1"

	cconf "qbox.us/cc/config"
	jsonlog "qbox.us/http/audit/jsonlog.v3"

	"qiniu.com/argus/argus/gate"
	"qiniu.com/argus/atserving/config"
)

func main() {

	runtime.GOMAXPROCS(runtime.NumCPU())

	var (
		xl  = xlog.NewWith("main")
		ctx = xlog.NewContext(context.Background(), xl)

		conf Config
	)

	cconf.Init("f", "argus-gate", "argus-gate.conf")
	if err := cconf.Load(&conf.FileConfig); err != nil {
		log.Fatal("Failed to load configure file!")
	}
	if err := InitConfig(ctx, &conf); err != nil {
		xl.Fatalf("init config failed. %v", err)
	}

	log.SetOutputLevel(conf.DebugLevel)
	log.Debugf("load conf %#v", conf)

	routers := gate.NewProxyRoutes()
	proxy := &gate.Proxy{
		Routes: routers,
	}
	method, _ := reflect.TypeOf(proxy).MethodByName("Do")
	handler, _ := rpcutil.HandlerCreator{}.New(reflect.ValueOf(proxy), method)

	{
		conf.ProxyRoutes.Register(
			func(k1, v1 interface{}) error {
				var k2 = k1.(string)
				if len(k2) == 0 {
					return nil
				}
				if v1 == nil {
					routers.Del(k2)
				} else {
					var v2 = v1.(gate.ProxyRoute)
					if strings.TrimSpace(v2.TLS.CertPem) != "" && strings.TrimSpace(v2.TLS.CertKey) != "" {
						pem, err := conf.FetchValue(ctx, v2.TLS.CertPem)
						if err == nil {
							v2.TLS.CertPem = string(pem)
						}
						key, err := conf.FetchValue(ctx, v2.TLS.CertKey)
						if err == nil {
							v2.TLS.CertKey = string(key)
						}
					}
					routers.Set(k2, v2)
				}
				return nil
			},
		)
		{
			for _, route := range conf.FileConfig.ProxyRoutes {
				routers.Set(route.Path, route)
			}
			ks, vs, _ := conf.ProxyRoutes.Values(ctx)
			for i, k1 := range ks {
				var k2 = k1.(string)
				if len(k2) == 0 {
					continue
				}
				var v2 = vs[i].(gate.ProxyRoute)
				if strings.TrimSpace(v2.TLS.CertPem) != "" && strings.TrimSpace(v2.TLS.CertKey) != "" {
					pem, err := conf.FetchValue(ctx, v2.TLS.CertPem)
					if err == nil {
						v2.TLS.CertPem = string(pem)
					}
					key, err := conf.FetchValue(ctx, v2.TLS.CertKey)
					if err == nil {
						v2.TLS.CertKey = string(key)
					}
				}

				routers.Set(k2, v2)
			}
		}
	}

	al, logf, err := jsonlog.Open("ARGUS-GATE", &conf.AuditLog, nil)
	if err != nil {
		log.Fatal("jsonlog.Open failed:", errors.Detail(err))
	}
	defer logf.Close()

	alMux := servestk.New(restrpc.NewServeMux(), al.Handler)
	alMux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("ok argus-gate"))
	})
	alMux.Handle("GET /metrics", promhttp.Handler())

	mux := http.NewServeMux()
	alMux.SetDefault(mux)

	mux.Handle("/", handler)

	if err := http.ListenAndServe(
		net.JoinHostPort("0.0.0.0", config.GetString(ctx, conf.HTTPPort, "80")),
		alMux,
	); err != nil {
		log.Errorf("argus start error: %v", err)
	}

	log.Error("shutdown...")
}
