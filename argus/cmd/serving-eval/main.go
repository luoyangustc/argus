package main

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	_ "net/http/pprof"
	"os"
	"path"
	"runtime"
	"strings"

	log "qiniupkg.com/x/log.v7"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	dto "github.com/prometheus/client_model/go"
	"github.com/prometheus/common/expfmt"

	"github.com/qiniu/errors"
	filelog "github.com/qiniu/filelog/log"
	restrpc "github.com/qiniu/http/restrpc.v1"
	servestk "github.com/qiniu/http/servestk.v1"
	xlog "github.com/qiniu/xlog.v1"

	cconf "qbox.us/cc/config"
	"qbox.us/dht"
	jsonlog "qbox.us/http/audit/jsonlog.v3"
	"qbox.us/net/httputil"

	"qiniu.com/argus/atserving/config"
	"qiniu.com/argus/atserving/model"
	"qiniu.com/argus/atserving/simplequeue/sq"
	"qiniu.com/argus/com/pprof"
	eval "qiniu.com/argus/serving_eval"
	"qiniu.com/argus/serving_eval/inference"
	STS "qiniu.com/argus/sts/client"
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
		cconf.Init("f", "serving-eval", "servering-eval.conf")
		if err := cconf.Load(&conf.FileConfig); err != nil {
			xl.Fatal("Failed to load configure file!")
		}
	}

	log.SetOutputLevel(conf.DebugLevel)
	if err := InitConfig(ctx, &conf); err != nil {
		xl.Fatalf("init config failed. %v", err)
	}

	xl.Info("loaded conf", dumps(conf))
	xl.Info("run mode", conf.RunMode, dumps(conf.Mode))

	// eval 的全局变量
	{
		if conf.RunMode != _ModeStandalone {
			{
				name, err := conf.AppName.Value(ctx)
				if err != nil {
					xl.Fatal("App Name required")
				}
				eval.APP = name.(string)
			}
			{
				version, err := conf.AppVersion.Value(ctx)
				if err != nil {
					xl.Fatal("App Version required")
				}
				eval.VERSION = version.(string)
			}
		}
	}

	// 创建 STS.Client
	var sts STS.Client
	if conf.Mode.OpenNsq {
		sts = func() STS.Client {
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
	}

	var workConfig1 model.ConfigWorker
	{
		var vv, err = conf.Worker.Value(ctx)
		xl.Warnf("%v", err)
		workConfig1 = vv.(model.ConfigWorker)
	}

	//  创建 eval.Handler
	var core eval.Core
	var handler eval.Handler
	{
		if conf.Mode.UseMockEval {
			core := eval.NewMockCore()
			handler = eval.NewHandler(core, conf.Workspace)
		} else {
			var vv, err = conf.AppRelease.Value(ctx)
			if err != nil {
				xl.Fatalf("get app release config failed. %v", err)
			}
			var releaseConfig = vv.(model.ConfigAppRelease)
			var workspace = path.Join(conf.Workspace, "init")
			os.Mkdir(workspace, 0644)
			var evalConfig = &eval.EvalConfig{
				App:          eval.APP,
				TarFile:      releaseConfig.TarFile,
				ImageWidth:   releaseConfig.ImageWidth,
				CustomFiles:  releaseConfig.CustomFiles,
				CustomValues: releaseConfig.CustomValues,
				Workspace:    workspace,
			}
			evalConfig.UseDevice = config.GetString(ctx, conf.UseDevice, "CPU")
			evalConfig.BatchSize = int(config.GetInt64(ctx, conf.BatchSize, 1))
			evalConfig.MaxConcurrent = int(workConfig1.MaxConcurrent)
			evalConfig.Fetcher.RsHost = config.GetString(ctx, conf.RsHost, "")
			{
				value1, err := conf.Owner.Value(ctx)
				if err == nil {
					value2 := value1.(model.Owner)
					evalConfig.Fetcher.UID = value2.UID
					evalConfig.Fetcher.AK = value2.AK
					evalConfig.Fetcher.SK = value2.SK
				}
			}
			if conf.Mode.OpenNsq {
				var vv, _ = conf.Worker.Value(ctx)
				var workConfig1 = vv.(model.ConfigWorker)
				evalConfig.PreProcess.On = true
				evalConfig.PreProcess.Threads = int(workConfig1.MaxConcurrent)
			}
			if tensorType := os.Getenv(eval.TensorTypeEnv); tensorType != "" {
				core, err = eval.NewTensorrtEvalNet(ctx, evalConfig, tensorType)
			} else {
				xl.Info("use INTEGRATE", config.GetString(ctx, conf.Integrate, _IntegratePython))
				switch config.GetString(ctx, conf.Integrate, _IntegratePython) {
				case _IntegrateLib:
					core, err = eval.NewInference(ctx, evalConfig, sts)
				case _IntegrateLib2:
					core, err = eval.NewInferenceDirect(
						ctx, evalConfig,
						inference.NewLib2(ctx, evalConfig.Workspace, "inference.so"),
					)
				case _IntegratePython:
					core, err = eval.NewEvalNet(ctx, evalConfig, sts)
				case _IntegrateZMQ:
					core, err = eval.NewInferenceDirect(
						ctx, evalConfig, inference.NewZmq(),
					)
				case _IntegrateGrpc:
					core, err = eval.NewInferenceDirect(
						ctx, evalConfig, inference.NewGrpc(),
					)
				case _IntegrateNative:
					core, err = eval.NewInferenceDirect(
						ctx, evalConfig, inference.NewNative(),
					)
				default:
					core, err = eval.NewEvalNet(ctx, evalConfig, sts)
				}
			}

			if err != nil {
				log.Fatal("Init eval net failed, err:", err)
			}
			handler = eval.NewHandler(core, conf.Workspace)
		}
	}

	// 创建 nsq.Consumer
	var consumer eval.Consumer
	if conf.Mode.OpenNsq {
		var workConfig = eval.WorkerConfig{
			MaxConcurrent: workConfig1.MaxConcurrent,
			Wait:          workConfig1.Deplay4Batch,
			BatchSize:     config.GetInt64(ctx, conf.BatchSize, 1),
		}

		worker := eval.NewWorker(
			workConfig, handler, sts,
			func(id int64, tr eval.TaskRequest) eval.Task { return eval.NewTask(id, tr) },
		)

		var vv2, _ = conf.SQD.Value(ctx)
		var sqd = vv2.([]model.ConfigConsumer)

		// TODO err
		ccs := func() []sq.ConsumerConfig {
			ccs := make([]sq.ConsumerConfig, 0, len(sqd))
			// var inFlight = int(config.GetInt64(ctx, conf.BatchSize, 1) * 3) // TODO
			var inFlight = int(workConfig1.MaxConcurrent)
			for _, c1 := range sqd {

				c2 := sq.NewConsumerConfig()
				c2.MaxInFlight = &inFlight
				c2.Addresses = c1.Addresses
				c2.Topic = "first_" + eval.APP + "_" + eval.VERSION
				c2.Channel = c2.Topic + "_0"
				ccs = append(ccs, c2)

				c3 := sq.NewConsumerConfig()
				c3.MaxInFlight = &inFlight
				c3.Addresses = c1.Addresses
				c3.Topic = "first_" + eval.APP
				c3.Channel = c3.Topic + "_0"
				ccs = append(ccs, c3)
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
		consumer, _ = eval.NewConsumer(worker, ccs)
		defer consumer.StopAndWait()
	}

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
			w.Write([]byte("ok " + eval.APP))
		})
		mux.HandleFunc("GET /metrics", func(w http.ResponseWriter, r *http.Request) {
			var reg = prometheus.DefaultGatherer
			if metrics, ok := core.(Metrics); ok {
				ms, _ := metrics.Metrics()
				var parser expfmt.TextParser
				parsed, err := parser.TextToMetricFamilies(strings.NewReader(ms))
				if err != nil {
				}
				var result []*dto.MetricFamily
				for _, mf := range parsed {
					result = append(result, mf)
				}
				reg = Gatherer{
					Gatherer: prometheus.DefaultGatherer,
					more:     result,
				}
			}
			promhttp.HandlerFor(reg, promhttp.HandlerOpts{}).ServeHTTP(w, r)
		})
		mux.HandleFunc("GET /configs", func(w http.ResponseWriter, r *http.Request) {
			m, err := config.DumpSimpleStruct(context.Background(), conf)
			if err != nil {
				httputil.ReplyErr(w, http.StatusInternalServerError, err.Error())
			} else {
				httputil.Reply(w, http.StatusOK, m)
			}
		})
		mustSerial := true
		if integrate := config.GetString(
			ctx, conf.Integrate, _IntegratePython,
		); integrate == _IntegrateLib2 ||
			integrate == _IntegrateZMQ ||
			integrate == _IntegrateGrpc {
			mustSerial = false
		}
		service := eval.NewService(handler, mustSerial)
		router := &restrpc.Router{
			PatternPrefix: "v1",
			Factory:       restrpc.Factory,
			Mux:           mux,
		}
		router.Register(service)

		{
			mux2 := http.NewServeMux()
			mux.SetDefault(mux2)
			pprof.Handle(mux2, "")
		}
	}
	xl.Info("ListenAndServe port ", config.GetString(ctx, conf.HTTPPort, "80"))
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

////////////////////////////////////////////////////////////////////////////////

type Metrics interface {
	Metrics() (string, error)
}

type Gatherer struct {
	prometheus.Gatherer
	more []*dto.MetricFamily
}

func (g Gatherer) Gather() ([]*dto.MetricFamily, error) {
	mfs, err := g.Gatherer.Gather()
	if err != nil {
		return mfs, err
	}
	return append(mfs, g.more...), nil
}

type MetricsResponseWriter struct {
	code   int
	header http.Header
	*bytes.Buffer
	io.WriteCloser
}

func (w *MetricsResponseWriter) Header() http.Header {
	return w.header
}
func (w *MetricsResponseWriter) Write(bs []byte) (int, error) {
	if w.WriteCloser != nil {
		return w.WriteCloser.Write(bs)
	}
	return w.Buffer.Write(bs)
}
func (w *MetricsResponseWriter) WriteHeader(code int) {
	w.code = code
}
