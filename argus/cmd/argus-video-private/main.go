package main

import (
	"context"
	"encoding/json"
	"net/http"
	"path"
	"runtime"

	"qiniu.com/argus/argus/video_proxy"
	"qiniu.com/argus/video"

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
	"qiniu.com/argus/serving_eval"
	OPS "qiniu.com/argus/video/ops"
	"qiniu.com/argus/video/segment"
	"qiniu.com/argus/video/vframe"
)

func init() {
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
		cconf.Init("f", "argus-video-private", "argus-video-private.conf")
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
		_video video.Video
		ops    video.OPs

		saverHook video.SaverHook
	)
	{
		OPS.RegisterPulp()
		OPS.RegisterTerror()
		OPS.RegisterTerrorComplex()
		OPS.RegisterPolitician()
		OPS.RegisterFaceDetect()
		OPS.RegisterFaceGroupSearch()
		OPS.RegisterImageLabel()

		OPS.RegisterTerrorClassify()
		OPS.RegisterTerrorDetect()
		OPS.RegisterDetection()
		OPS.RegisterMagicearTrophy()
		OPS.RegisterFaceGroupPrivateSearch()
	}

	{
		m := make(map[string]video.OPConfig)
		val, _ := conf.OPs.Value(ctx)
		for k, v := range val.(map[string]video.OPConfig) {
			m[k] = v
		}
		ops = video.NewOPs(m)
	}

	{
		var v1, _ = conf.Segment.Value(ctx)
		var v2, _ = conf.Vframe.Value(ctx)
		// var segment = video.NewSegment(v1.(video.SegmentParams))
		_vframe := vframe.NewVframeBase64(
			vframe.NewVframe(
				vframe.VframeConfig{
					Dir: path.Join(conf.Workspace, "data"),
				},
				uriProxy,
			))
		_segment := segment.NewSegment(
			segment.SegmentConfig{
				Dir: path.Join(conf.Workspace, "data"),
			},
			uriProxy,
		)
		// _video = video.NewVideo(
		// 	_vframe, v2.(vframe.VframeParams),
		// 	_segment, v1.(segment.SegmentParams),
		// )
		_video = _Video{
			Vframe:        _vframe,
			VframeParams:  v2.(vframe.VframeParams),
			Segment:       _segment,
			SegmentParams: v1.(segment.SegmentParams),
		}
	}

	{
		if len(conf.Savespace) == 0 {
			xl.Infof("load savespace config failed.")
		} else {
			saveCfg := video.FileSaveConfig{
				SaveSpace: conf.Savespace,
			}
			saverHook = video.NewFileSaver(saveCfg)
		}
	}
	var service = video.NewService(ctx, _video, ops, nil, saverHook)
	var proxy = video_proxy.NewProxy(conf.FileConfig.Host)

	var mux *servestk.ServeStack
	{
		// 审计日志
		al, logf, err := jsonlog.Open("ARGUS_VFRAME", &conf.AuditLog, nil)
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
		router.Register(proxy)
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

type _Video struct {
	vframe.Vframe
	vframe.VframeParams

	segment.Segment
	segment.SegmentParams
}

func (v _Video) Run(
	ctx context.Context,
	req video.VideoRequest,
	ops map[string]video.OP,
	saverOPHook video.SaverOPHook,
	ends func(string) video.EndHook,
	cuts func(string) video.CutHook,
	segments func(string) video.SegmentHook, // TODO
) error {

	return nil // 不再维护
	// return svideo.VideoRunner{
	// 	CutRunner: svideo.VideoCutRunnerV1{Vframe: v.Vframe}.Run,
	// 	ProcessFactory: svideo.NewProcessFactoryV1(
	// 		ctx, ops, saverOPHook, true, ends, segments, cuts),
	// }.Run(ctx, req)
}

//----------------------------------------------------------------------------//
type fileLog struct {
	*filelog.Logger
}

func (l fileLog) Output(calldepth int, s string) error {
	return l.Log([]byte(s))
}
