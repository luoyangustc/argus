package argus_live

import (
	"context"
	"encoding/json"
	"net/http"
	"path"
	"runtime"

	"github.com/prometheus/client_golang/prometheus/promhttp"

	"github.com/qiniu/errors"
	filelog "github.com/qiniu/filelog/log"
	httputil "github.com/qiniu/http/httputil.v1"
	restrpc "github.com/qiniu/http/restrpc.v1"
	servestk "github.com/qiniu/http/servestk.v1"
	log "github.com/qiniu/log.v1"
	xlog "github.com/qiniu/xlog.v1"

	cconf "qbox.us/cc/config"
	jsonlog "qbox.us/http/audit/jsonlog.v3"

	"qiniu.com/argus/atserving/config"
	eval "qiniu.com/argus/serving_eval"
	"qiniu.com/argus/video"
	"qiniu.com/argus/video/segment"
	"qiniu.com/argus/video/vframe"
)

func init() {
}

func Main() {

	runtime.GOMAXPROCS(runtime.NumCPU())

	var (
		xl   = xlog.NewWith("main")
		ctx  = xlog.NewContext(context.Background(), xl)
		conf Config
	)

	// 加载配置
	{
		cconf.Init("f", "argus-live-private", "argus-live-private.conf")
		if err := cconf.Load(&conf.FileConfig); err != nil {
			xl.Fatal("Failed to load configure file!")
		}
	}

	log.SetOutputLevel(conf.DebugLevel)
	if err := InitConfig(ctx, &conf); err != nil {
		xl.Fatalf("init config failed. %v", err)
	}

	xl.Info("loaded conf", dumps(conf))

	var (
		uriProxy vframe.URIProxy = vframe.NewURIProxy(
			"http://127.0.0.1:" + config.GetString(ctx, conf.HTTPPort, "80") + "/uri",
		)
		_video    video.Video
		ops       video.OPs
		jobs      video.Jobs
		worker    video.Worker
		saverHook video.SaverHook
	)

	{
		m := make(map[string]video.OPConfig)
		if conf.OPs != nil {
			m = conf.OPs
		}
		ops = video.NewOPs(m)
	}

	{
		// _vframe := vframe.NewVframeBase64(
		// 	vframe.NewLive(
		// 		vframe.VframeConfig{
		// 			Dir: path.Join(conf.Workspace, "data"),
		// 		},
		// 		uriProxy,
		// 	),
		// )
		_segment := segment.NewSegment(
			segment.SegmentConfig{
				Dir: path.Join(conf.Workspace, "data"),
			},
			uriProxy,
		)
		// _video = video.NewLive(
		// 	_vframe, conf.Vframe,
		// 	_segment, conf.Segment,
		// )
		_video = _Live{
			// VideoCutRunnerFunc: svideo.VideoCutRunnerV2{}.Run,
			VframeParams:  conf.Vframe,
			Segment:       _segment,
			SegmentParams: conf.Segment,
		}
	}

	{
		var v, _ = conf.JobsInMgoHost.Value(ctx)
		conf.FileConfig.Jobs.Mgo.Mgo.Host = v.(string)
		var err error
		if jobs, err = video.NewJobsInMgo(conf.FileConfig.Jobs.Mgo); err != nil {
			log.Fatal("failed to create jobsInMgo:", errors.Detail(err))
		}

	}

	{
		if len(conf.SaveConfig.SaveSpace) == 0 {
			xl.Infof("load savespace config failed.")
		} else {
			saverHook = video.NewFileSaver(conf.SaveConfig)
		}
	}

	var service = video.NewServiceLive(ctx, _video, ops, jobs, saverHook, conf.LiveTimeout)
	{
		var v, _ = conf.Worker.Value(ctx)
		worker = video.NewLiveWorker(v.(video.WorkerConfig), jobs, _video, ops, saverHook, service)
		go worker.Run()
	}

	var mux *servestk.ServeStack
	{
		// 审计日志
		al, logf, err := jsonlog.Open("ARGUS_LIVE", &conf.AuditLog, nil)
		if err != nil {
			log.Fatal("jsonlog.Open failed:", errors.Detail(err))
		}
		defer logf.Close()

		// run Service
		mux = servestk.New(restrpc.NewServeMux(), func(
			w http.ResponseWriter, req *http.Request, f func(http.ResponseWriter, *http.Request)) {
			req.Header.Set("Authorization", "QiniuStub uid=1&ut=0")
			f(w, req)
		}, al.Handler)
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

		router := &restrpc.Router{
			PatternPrefix: "v1",
			Factory:       restrpc.Factory,
			Mux:           mux,
		}
		router.Register(service)
		{
			router := &restrpc.Router{
				PatternPrefix: "",
				Factory:       restrpc.Factory,
				Mux:           mux,
			}
			router.Register(uriProxy)
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

type _Live struct {
	// svideo.VideoCutRunnerFunc
	vframe.VframeParams

	segment.Segment
	segment.SegmentParams
}

func (v _Live) Run(
	ctx context.Context,
	req video.VideoRequest,
	ops map[string]video.OP,
	saverOPHook video.SaverOPHook,
	ends func(string) video.EndHook,
	cuts func(string) video.CutHook,
	segments func(string) video.SegmentHook, // TODO
) error {

	// return svideo.VideoRunner{
	// 	CutRunner: svideo.VideoCutRunnerV1{Vframe: v.Vframe}.Run,
	// 	NewProcessFactory: svideo.NewProcessFactoryV1(
	// 		ctx, ops, saverOPHook, false, ends, segments, cuts),
	// }.Run(ctx, req, ops)
	return nil // 不再维护
	// return svideo.VideoRunner{
	// 	CutRunner: v.VideoCutRunnerFunc,
	// 	ProcessFactory: svideo.NewProcessFactoryV1(
	// 		ctx, ops, saverOPHook, false, ends, segments, cuts),
	// }.Run(ctx, req)

}

//----------------------------------------------------------------------------//
type fileLog struct {
	*filelog.Logger
}

func (l fileLog) Output(calldepth int, s string) error {
	return l.Log([]byte(s))
}
