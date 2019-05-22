package main

import (
	"context"
	"encoding/json"
	"net/http"
	"os"
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
	jsonlog "qbox.us/http/audit/jsonlog.v3"

	"qiniu.com/argus/atserving/config"
	"qiniu.com/argus/atserving/simplequeue/sq"
	"qiniu.com/argus/serving_eval"
	"qiniu.com/argus/video"
	OPS "qiniu.com/argus/video/ops"
	"qiniu.com/argus/video/segment"
	"qiniu.com/argus/video/vframe"
)

var (
	SERVING_GATE_HOST string
)

func init() {
	SERVING_GATE_HOST = os.Getenv("SERVING_GATE_HOST")
}

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
		// if strings.TrimSpace(SERVING_GATE_HOST) != "" {
		// 	conf.ArgVideo.ServingHost = SERVING_GATE_HOST
		// }
	}

	log.SetOutputLevel(conf.DebugLevel)
	if err := InitConfig(ctx, &conf); err != nil {
		xl.Fatalf("init config failed. %v", err)
	}

	xl.Info("loaded conf", dumps(conf))

	var (
		vProducer *vframe.Producer
		sProducer *segment.Producer
		_video    video.Video
		ops       video.OPs

		jobs   video.Jobs
		worker video.Worker

		saverHook video.SaverHook
	)
	{
		OPS.RegisterPulp()
		OPS.RegisterTerror()
		OPS.RegisterPolitician()
		OPS.RegisterFaceDetect()
		OPS.RegisterFaceGroupSearch()
		OPS.RegisterImageLabel()

		OPS.RegisterTerrorClassify()
		OPS.RegisterTerrorDetect()
		OPS.RegisterDetection()
		OPS.RegisterMagicearTrophy()
	}

	{
		m := make(map[string]video.OPConfig)
		ks, vs, _ := conf.OPs.Values(ctx)
		for i, k1 := range ks {
			m[k1.(string)] = vs[i].(video.OPConfig)
		}
		ops = video.NewOPs(m)
		conf.OPs.Register(
			func(k1, v1 interface{}) error {
				var name = k1.(string)
				if v1 == nil {
					ops.ResetOP(name, nil)
				} else {
					var v2 = v1.(video.OPConfig)
					ops.ResetOP(name, &v2)
				}
				return nil
			},
		)
	}

	{
		var sqHosts []string
		if vv, err := conf.SQHosts.Value(ctx); err != nil {
			xl.Fatalf("load sq hosts failed. %v", err)
		} else {
			sqHosts = vv.([]string)
		}

		vProducer = func() *vframe.Producer {
			var ps = make([]*sq.Producer, 0, len(sqHosts))
			for _, host := range sqHosts {
				p, _ := sq.NewProducer(host) // TODO err
				ps = append(ps, p)
			}
			return vframe.NewProducer(
				ps,
				"http://"+config.GetString(ctx, conf.CallBackHost, "")+"/cuts",
			) // TODO
		}()

		sProducer = func() *segment.Producer {
			var ps = make([]*sq.Producer, 0, len(sqHosts))
			for _, host := range sqHosts {
				p, _ := sq.NewProducer(host) // TODO err
				ps = append(ps, p)
			}
			return segment.NewProducer(
				ps,
				"http://"+config.GetString(ctx, conf.CallBackHost, "")+"/clips",
			) // TODO
		}()
	}

	{
		var v1, _ = conf.Segment.Value(ctx)
		var v2, _ = conf.Vframe.Value(ctx)
		// var segment = video.NewSegment(v1.(video.SegmentParams))
		_video = video.NewVideo(
			vProducer, v2.(vframe.VframeParams),
			sProducer, v1.(segment.SegmentParams),
		)
	}

	{
		var v, _ = conf.JobsInMgoHost.Value(ctx)
		conf.FileConfig.Jobs.Mgo.Mgo.Host = v.(string)
		jobs, _ = video.NewJobsInMgo(conf.FileConfig.Jobs.Mgo)
	}

	{
		if v, err := conf.SaveConfig.Value(ctx); err != nil {
			xl.Infof("load save config failed. %v", err)
		} else {
			saveCfg := v.(video.KodoSaveConfig)
			saverHook = video.NewKodoSaver(saveCfg)
		}
	}

	{
		var v, _ = conf.Worker.Value(ctx)
		worker = video.NewWorker(v.(video.WorkerConfig), jobs, _video, ops, saverHook)
		go worker.Run()
	}

	var service = video.NewService(ctx, _video, ops, jobs, saverHook)
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
				PatternPrefix: "cuts",
				Factory:       restrpc.Factory,
				Mux:           mux,
			}
			router.Register(vProducer)
		}
		{
			router := &restrpc.Router{
				PatternPrefix: "clips",
				Factory:       restrpc.Factory,
				Mux:           mux,
			}
			router.Register(sProducer)
		}

		router := &restrpc.Router{
			PatternPrefix: "v1",
			Factory:       restrpc.Factory,
			Mux:           mux,
		}
		router.Register(service)
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
