package main

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"runtime"
	"strconv"
	"time"

	"qiniu.com/argus/AIProjects/wangan/yuqing"

	restrpc "github.com/qiniu/http/restrpc.v1"
	servestk "github.com/qiniu/http/servestk.v1"
	"github.com/qiniu/log.v1"
	cconf "qbox.us/cc/config"
	"qbox.us/errors"
	jsonlog "qbox.us/http/audit/jsonlog.v3"
	"qiniu.com/argus/AIProjects/wangan/yuqing/fetcher"
	"qiniu.com/argus/AIProjects/wangan/yuqing/kmq"
	xlog "qiniupkg.com/x/xlog.v7"
)

const (
	access_key = "" // kodo admin ak
	secret_key = "" // kodo admin sk
)

type Config struct {
	HTTPPort      int                `json:"http_port"`
	DebugLevel    int                `json:"debug_level"`
	Auditlog      jsonlog.Config     `json:"audit_log"`
	Kodo          fetcher.KodoConfig `json:"kodo"`
	IPNets        string             `json:"ip_nets"`
	UserWhiteList []int              `json:"user_whitelist"`
	UserBlackList []uint32           `json:"user_blacklist"`
	Output        string             `json:"output"`
	UID           uint32             `json:"uid"`
}

func main() {
	runtime.GOMAXPROCS(runtime.NumCPU())
	var (
		conf Config
		ctx  = context.Background()
		xl   = xlog.New("main")
		f    *os.File
		err  error
	)
	cconf.Init("f", "yuqing_fetcher", "yuqing_fetcher")
	if err := cconf.Load(&conf); err != nil {
		log.Fatalf("Failed to load configure file, error: %v", err)
	}
	log.SetOutputLevel(conf.DebugLevel)
	log.Infof("load conf %v", conf)

	al, logf, err := jsonlog.Open("WANGAN-GATE", &conf.Auditlog, nil)
	if err != nil {
		log.Fatal("jsonlog Open failed:", errors.Detail(err))
	}
	defer logf.Close()
	if conf.Kodo.AccessKey == "" {
		conf.Kodo.AccessKey = access_key
	}
	if conf.Kodo.SecretKey == "" {
		conf.Kodo.SecretKey = secret_key
	}

	if conf.Output != "" {
		f, err = os.OpenFile(conf.Output, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0644)
		if err != nil {
			log.Fatalf("fail to create output file %s, err: %s", conf.Output, err.Error())
		}
		defer f.Close()
	}

	kodo, err := fetcher.NewKodo(conf.Kodo)
	if err != nil {
		log.Fatal("fail to create kodo error:", err.Error())
	}

	for _, uid := range conf.UserWhiteList {
		if err = kodo.AddItbl(ctx, uid, nil); err != nil {
			log.Fatalf("fail to get %d itbls", uid)
		}
	}

	for _, uid := range conf.UserBlackList {
		if err = kodo.AddBlackList(ctx, uid, []string{}); err != nil {
			log.Fatalf("fail to get blacklist %d, error: %s", uid, err.Error())
		}
	}

	for _, bucket := range kodo.Buckets() {
		xl.Infof("buckets: %v", bucket)
	}

	for _, bucket := range kodo.BlackList() {
		xl.Infof("BlackList: %v", bucket)
	}

	go kodo.Run(ctx)

	go func() {
		t := 0
		for {
			t = t + 5
			time.Sleep(time.Second * 5)
			m := kodo.GetMetrics()
			xl.Infof("Metrics: total(%d) video(%d) shanghai(%d) inner(%d) available(%d) unavailable(%d) total_qps(%f) qps(%f)",
				m.Total, m.TargetVideo, m.TargetRegion, m.Inner, m.TargetUser, m.Unavailable, float64(m.Total)/float64(t), float64(m.TargetUser)/float64(t))
		}
	}()

	kmq := kmq.NewKMQ(conf.Kodo.Config, conf.UID)

	go func() {
		for {
			select {
			case msg := <-kodo.Message():
				//xl.Infof("get message %v", msg)
				err := kmq.Produce(ctx, msg, yuqing.SourceTypeMiaopai, yuqing.MediaTypeVideo)
				if f != nil {
					f.WriteString(fmt.Sprintf("Video: %v, err: %v\n", msg, err))
				}
			}
		}
	}()
	server := fetcher.NewServer(kodo)

	mux := servestk.New(restrpc.NewServeMux(), func(w http.ResponseWriter, req *http.Request, f func(http.ResponseWriter, *http.Request)) {
		req.Header.Set("Authorization", "QiniuStub uid=1&ut=0")
		f(w, req)
	}, al.Handler)
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("ok " + "wanga-gate"))
	})
	router := &restrpc.Router{
		PatternPrefix: "v1",
		Factory:       restrpc.Factory,
		Mux:           mux,
	}
	router.Register(server)
	if conf.HTTPPort <= 0 {
		conf.HTTPPort = 80
	}
	if err := http.ListenAndServe(
		"0.0.0.0:"+strconv.Itoa(conf.HTTPPort), mux,
	); err != nil {
		log.Fatal("http.ListenAndServe", err)
	}
}
