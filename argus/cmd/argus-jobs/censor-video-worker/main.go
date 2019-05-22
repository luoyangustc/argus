package main

import (
	"context"
	"net"
	"net/http"
	"path"
	"runtime"

	"github.com/prometheus/client_golang/prometheus/promhttp"

	cconf "qbox.us/cc/config"
	"qbox.us/dht"
	jsonlog "qbox.us/http/audit/jsonlog.v3"

	"github.com/qiniu/errors"
	filelog "github.com/qiniu/filelog/log"
	"github.com/qiniu/http/httputil.v1"
	"github.com/qiniu/http/restrpc.v1"
	"github.com/qiniu/http/servestk.v1"
	"github.com/qiniu/log.v1"
	"github.com/qiniu/xlog.v1"

	"qiniu.com/argus/atserving/config"
	"qiniu.com/argus/atserving/model"
	"qiniu.com/argus/atserving/simplequeue/sq"
	"qiniu.com/argus/bjob/job"
	"qiniu.com/argus/bjob/workers"
	CensorWorkers "qiniu.com/argus/bjob/workers/censor"
	"qiniu.com/argus/com/pprof"
	STS "qiniu.com/argus/sts/client"
	"qiniu.com/argus/video"
	OPS "qiniu.com/argus/video/ops"
	"qiniu.com/argus/video/segment"
	"qiniu.com/argus/video/vframe"
)

func main() {

	runtime.GOMAXPROCS(runtime.NumCPU())

	var (
		xl  = xlog.NewWith("main")
		ctx = xlog.NewContext(context.Background(), xl)
	)
	cconf.Init("f", "inference-image-worker", "inference-image-worker.conf")

	var conf = Config{}
	if err := cconf.Load(&conf.FileConfig); err != nil {
		xl.Fatal("Failed to load configure file!")
	}
	if err := InitConfig(ctx, &conf); err != nil {
		xl.Fatalf("init config failed. %v", err)
	}
	xl.Infof("load conf %#v", conf)

	al, logf, err := jsonlog.Open("W", &conf.FileConfig.AuditLog, nil)
	if err != nil {
		log.Fatal("jsonlog.Open failed:", errors.Detail(err))
	}
	defer logf.Close()

	mux := servestk.New(restrpc.NewServeMux(), al.Handler)
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("ok serving-gate"))
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
		mux2 := http.NewServeMux()
		mux.SetDefault(mux2)
		pprof.Handle(mux2, "")
	}

	var workerConfig job.WorkerNodeConfig
	if vv, err := conf.Worker.Value(ctx); err != nil {
		xl.Fatalf("load worker config failed. %v", err)
	} else {
		workerConfig = vv.(job.WorkerNodeConfig)
	}
	var iiWorkerConfig workers.InferenceImageConfig
	xl.Info("load conf", config.DumpJsonConfig(iiWorkerConfig), config.DumpJsonConfig(conf))

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

	var (
		_vframe   vframe.Vframe
		_video    video.Video
		ops       video.OPs
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

	var proxy = vframe.NewSTSProxy(
		"http://127.0.0.1:"+config.GetString(ctx, conf.HTTPPort, "80")+"/uri",
		sts,
	)

	_vframe = vframe.NewVframeBase64(
		vframe.NewVframe(
			vframe.VframeConfig{
				Dir: path.Join(conf.Workspace, "data"),
			},
			proxy,
		),
	)

	{
		var v1, _ = conf.Segment.Value(ctx)
		var v2, _ = conf.Vframe.Value(ctx)
		// var segment = video.NewSegment(v1.(video.SegmentParams))
		_video = video.NewVideo(
			_vframe, v2.(vframe.VframeParams),
			nil, v1.(segment.SegmentParams),
		)
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
		sq.Logger = fileLog{Logger: lg}

		cvWorker := CensorWorkers.NewVideoWorker(
			workers.NewInferenceVideoWorker(iiWorkerConfig, _video, ops, saverHook))
		_, _ = job.NewAsyncS(
			func() []sq.ConsumerConfig {
				return []sq.ConsumerConfig{
					func() sq.ConsumerConfig {
						c1 := sq.NewConsumerConfig()

						var sqHosts []model.ConfigConsumer
						if vv, err := conf.SQHosts.Value(ctx); err != nil {
							xl.Fatalf("load sq hosts failed. %v", err)
						} else {
							sqHosts = vv.([]model.ConfigConsumer)
						}
						for _, v := range sqHosts {
							c1.Addresses = append(c1.Addresses, v.Addresses...)
						}

						//c1.MaxInFlight = &conf.SearchWorker.SQ.MaxInFlight
						c1.Topic = "first_argus_jobw_" + JOB_WORKER
						c1.Channel = c1.Topic + "_0"

						if workerConfig.MaxInFlight > 0 {
							c1.MaxInFlight = &workerConfig.MaxInFlight
						} else {
							var maxInFlight = 100
							c1.MaxInFlight = &maxInFlight
						}
						return c1
					}(),
				}
			}(),
			func(ctx context.Context, req []byte) ([]byte, error) {
				return cvWorker.Do(ctx, job.NewTask(req))
			},
		)
	}

	{
		router := &restrpc.Router{
			PatternPrefix: "",
			Factory:       restrpc.Factory,
			Mux:           mux,
		}
		router.Register(proxy)
	}

	if err = http.ListenAndServe(
		net.JoinHostPort("0.0.0.0",
			config.GetString(ctx, conf.HTTPPort, "80")),
		mux); err != nil {
		log.Fatal("http.ListenAndServe", err)
	}
}

type fileLog struct {
	*filelog.Logger
}

func (l fileLog) Output(calldepth int, s string) error {
	return l.Log([]byte(s))
}
