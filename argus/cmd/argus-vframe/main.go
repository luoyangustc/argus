package main

import (
	"context"
	"encoding/json"
	"net/http"
	"path"
	"runtime"

	"github.com/prometheus/client_golang/prometheus/promhttp"

	"github.com/qiniu/errors"
	filelog "github.com/qiniu/filelog/log"
	"github.com/qiniu/http/httputil.v1"
	"github.com/qiniu/http/restrpc.v1"
	"github.com/qiniu/http/servestk.v1"
	"github.com/qiniu/log.v1"
	"github.com/qiniu/xlog.v1"

	cconf "qbox.us/cc/config"
	"qbox.us/dht"
	jsonlog "qbox.us/http/audit/jsonlog.v3"

	"qiniu.com/argus/atserving/config"
	"qiniu.com/argus/atserving/model"
	"qiniu.com/argus/atserving/simplequeue/sq"
	"qiniu.com/argus/serving_eval"
	STS "qiniu.com/argus/sts/client"
	"qiniu.com/argus/video/vframe"
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
		cconf.Init("f", "argus-vframe", "argus-vframe.conf")
		if err := cconf.Load(&conf.FileConfig); err != nil {
			xl.Fatal("Failed to load configure file!")
		}
	}

	log.SetOutputLevel(conf.DebugLevel)
	if err := InitConfig(ctx, &conf); err != nil {
		xl.Fatalf("init config failed. %v", err)
	}

	xl.Info("loaded conf", dumps(conf))

	// 创建 STS.Client
	var sts = func() STS.Client {
		var vv, _ = conf.STSHosts.Value(ctx)
		var hosts = vv.([]model.ConfigSTSHost)
		var nodes = make(dht.NodeInfos, 0, len(hosts))
		for _, host := range hosts {
			nodes = append(nodes, dht.NodeInfo{Host: host.Host, Key: []byte(host.Key)})
		}
		lb, update := STS.NewLB(nodes, func() string { return xlog.GenReqId() }, nil)
		conf.STSHosts.Register(
			func(v interface{}) error {
				if v == nil {
					return nil
				}
				xl.Infof("sts update %#v", v)
				hosts := v.([]model.ConfigSTSHost)
				var nodes = make(dht.NodeInfos, 0, len(hosts))
				for _, host := range hosts {
					nodes = append(nodes, dht.NodeInfo{Host: host.Host, Key: []byte(host.Key)})
				}
				update(context.Background(), nodes)
				return nil
			},
		)
		return lb
	}()

	// var proxy = vframe.NewURIProxy(
	// 	"http://127.0.0.1:" + config.GetString(ctx, conf.HTTPPort, "80") + "/uri",
	// )
	var proxy = vframe.NewSTSProxy(
		"http://127.0.0.1:"+config.GetString(ctx, conf.HTTPPort, "80")+"/uri",
		sts,
	)
	var vf = vframe.NewVframe(
		vframe.VframeConfig{
			Dir: path.Join(conf.Workspace, "data"),
		},
		proxy,
	)

	// 创建 nsq.Consumer
	var consumer eval.Consumer
	{
		var vv2, _ = conf.SQD.Value(ctx)
		var sqd = vv2.([]model.ConfigConsumer)

		// TODO err
		ccs := func() []sq.ConsumerConfig {
			ccs := make([]sq.ConsumerConfig, 0, len(sqd))
			for _, c1 := range sqd {
				c2 := sq.NewConsumerConfig()
				// c2.MaxInFlight = c1.MaxInFlight
				c2.Addresses = c1.Addresses
				c2.Topic = "first_argus_vframe" // TODO
				c2.Channel = c2.Topic + "_0"
				ccs = append(ccs, c2)
			}
			return ccs
		}()
		lg, err := filelog.NewLogger(
			conf.SQLog.LogDir,
			conf.SQLog.LogPrefix,
			conf.SQLog.TimeMode,
			conf.SQLog.ChunkBits,
		)
		if err != nil {
			xl.Fatalf("filelog.Open failed. %#v %v", conf.SQLog, err)
		}
		defer lg.Close()
		sq.Logger = fileLog{lg}
		consumer, _ = vframe.NewConsumer(vf, sts, ccs)
		defer consumer.StopAndWait()
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
				PatternPrefix: "",
				Factory:       restrpc.Factory,
				Mux:           mux,
			}
			router.Register(proxy)
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

//----------------------------------------------------------------------------//
type fileLog struct {
	*filelog.Logger
}

func (l fileLog) Output(calldepth int, s string) error {
	return l.Log([]byte(s))
}
