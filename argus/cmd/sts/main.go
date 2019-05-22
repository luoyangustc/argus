package main

import (
	"net/http"
	"os"
	"runtime"
	"time"

	"github.com/prometheus/client_golang/prometheus/promhttp"

	"github.com/qiniu/errors"
	"github.com/qiniu/http/restrpc.v1"
	"github.com/qiniu/http/servestk.v1"
	"github.com/qiniu/log.v1"

	"qbox.us/cc/config"
	jsonlog "qbox.us/http/audit/jsonlog.v3"

	"qiniu.com/argus/com/uri"
	"qiniu.com/argus/sts"
)

type Config struct {
	HTTPHost string            `json:"http_host"`
	AuditLog jsonlog.Config    `json:"audit_log"`
	Storage  sts.StorageConfig `json:"storage"`
	Fetch    struct {
		Qiniu uri.QiniuAdminHandlerConfig `json:"qiniu"`
	} `json:"fetch"`
}

func main() {

	config.Init("f", "sts", "sts.conf")
	var conf Config
	if err := config.Load(&conf); err != nil {
		log.Fatal("Failed to load configure file!")
	}
	log.Infof("load conf %#v", conf)

	conf.Storage.Overdue = conf.Storage.Overdue * time.Second

	//if os.Getenv("USE_MOCK") == "true" {}

	runtime.GOMAXPROCS(runtime.NumCPU()) // TODO

	var handler uri.Handler
	if os.Getenv("DEBUG") == "true" {
		handler = uri.New(
			uri.WithSimpleHTTPHandler(
				&http.Client{
					Transport: &http.Transport{
						ResponseHeaderTimeout: time.Second * 5,
					},
				},
			),
		)
	} else {
		handler = uri.New(
			uri.WithAdminAkSkV2(conf.Fetch.Qiniu,
				func() http.RoundTripper {
					return uri.StaticHeader{
						Header: map[string][]string{"X-From-Cdn": []string{"atlab"}},
						RT:     http.DefaultTransport,
					}
				}(),
			),
			uri.WithSimpleHTTPHandler(
				&http.Client{
					Transport: &http.Transport{
						ResponseHeaderTimeout: time.Second * 5,
					},
				},
			),
		)
	}

	var (
		fetcher sts.Fetcher = sts.NewFetcher(handler)
	)

	stg, err := sts.NewStorage(conf.Storage, conf.Storage.InitSlots(fetcher), conf.Storage.NewSlot(fetcher))
	if err != nil {
		log.Fatal("storage.New:", errors.Detail(err))
	}
	server := sts.NewServer(stg, fetcher)

	al, logf, err := jsonlog.Open("STS", &conf.AuditLog, nil)
	if err != nil {
		log.Fatal("jsonlog.Open failed:", errors.Detail(err))
	}
	defer logf.Close()

	mux := servestk.New(restrpc.NewServeMux(), al.Handler)
	mux.Handle("GET /metrics", promhttp.Handler())
	router := &restrpc.Router{
		PatternPrefix: "v1",
		Factory:       restrpc.Factory,
		Mux:           mux,
	}
	router.Register(server)

	if err = http.ListenAndServe(conf.HTTPHost, mux); err != nil {
		log.Fatal("http.ListenAndServe", err)
	}
}
