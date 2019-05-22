package video

import (
	"context"
	"encoding/json"
	"net/http"
	"path"
	"sync"

	httptransport "github.com/go-kit/kit/transport/http"
	"github.com/gorilla/mux"

	httputil "github.com/qiniu/http/httputil.v1"
	restrpc "github.com/qiniu/http/restrpc.v1"
	xlog "github.com/qiniu/xlog.v1"

	"qiniu.com/argus/service/middleware"
	scenario "qiniu.com/argus/service/scenario/video"
	. "qiniu.com/argus/service/service"
	svideo "qiniu.com/argus/service/service/video"
	"qiniu.com/argus/service/service/video/vod"
	"qiniu.com/argus/service/transport"
	"qiniu.com/argus/video"
	"qiniu.com/argus/video/vframe"
)

var (
	_DOC_SYNC_VIDEO = scenario.APIDoc{
		Name:    `同步视频任务`,
		Version: "1.0.0",
		Desc:    []string{`同步视频分析`},
		Request: `POST /v1/video/<vid>
Content-Type: application/json

{
	"data": {
		"uri": "http://xx"
	},
	"params": {
		"vframe": {
			"mode": <mode:int>,
			"interval": <interval:float>
		}
	},
	"ops": [
		{
			"op": <op:string>,
			"params": {
				"labels": [
					{
						"label": <label:string>,
						"select": <select:int>,
						"score": <score:float>
					},
					...
				],
				"terminate": {
					"mode": <mode:int>,
					"labels": {
						<label>: <max:int>
					}
				}
			}
		},
		... 
	]
}`,
		Response: `HTTP/1.1 200 OK
Content-Type: application/json

{
	<op:string>: {
		"labels": [
			{
				"label": <label:string>,
				"score": <score:float>
			},
			...
		],
		"segments": [
			{
				"offset_begin": <offset:int>,
				"offset_end": <offset:int>,
				"labels": [
					{
						"label": <label:string>,
						"score": <score:float>
					},
					...
				],
				"cuts": [
					{
						"offset": <offset:int>,
						"result": {}
					},
					...
				]   
			},
			...
		]
	}
}`,
		RequestParam: []scenario.APIDocParam{
			{Name: "vid", Type: "string", Must: "Yes", Desc: `视频唯一标识`},
			{Name: "data.uri", Type: "string", Must: "Yes", Desc: `视频地址`},
			{Name: "params.vframe.mode", Type: "int", Must: "No", Desc: `截帧逻辑，可选值为[0, 1]。0表示每隔固定时间截一帧，固定时间在vframe.interval中设定；1表示截关键帧。默认值为0`},
			{Name: "params.vframe.interval", Type: "float", Must: "No", Desc: `当params.vframe.mode取0时，用来设置每隔多长时间截一帧，单位s, 不填则取默认值5s`},
			{Name: "ops.[].op", Type: "string", Must: "Yes", Desc: `执行的推理cmd`},
			{Name: "ops.[].params.labels.[].label", Type: "string", Must: "No", Desc: `选择类别名，跟具体推理cmd有关`},
			{Name: "ops.[].params.labels.[].select", Type: "int", Must: "No", Desc: `类别选择条件，1表示忽略不选；2表示只选该类别`},
			{Name: "ops.[].params.labels.[].score", Type: "float", Must: "No", Desc: `类别选择的可信度参数，当select=1时表示忽略不选小于score的结果，当select=2时表示只选大于等于该值的结果`},
			{Name: "ops.[].params.terminate.mode", Type: "int", Must: "No", Desc: `提前退出类型。1表示按帧计数；2表示按片段计数`},
			{Name: "ops.[].params.terminate.label.max", Type: "int", Must: "No", Desc: `该类别的最大个数，达到该阈值则处理过程退出`},
		},
		ResponseParam: []scenario.APIDocParam{
			{Name: "op", Type: "string", Desc: `推理cmd`},
			{Name: "op.labels.[].label", Type: "string", Desc: `视频的判定结果`},
			{Name: "op.labels.[].score", Type: "float", Desc: `视频的判定结果可信度`},
			{Name: "op.segments.[].offset_begin", Type: "int", Desc: `片段起始的时间位置`},
			{Name: "op.segments.[].offset_end", Type: "int", Desc: `片段结束的时间位置`},
			{Name: "op.segments.[].labels.[].label", Type: "string", Desc: `视频的片段的判定结果，terror中{0:表示正常，1:表示暴恐}，pulp中{0:色情图片，1:性感图片，2:正常图片}, politician中{空或无值：未确定有政治人物，有值:有政治人物}`},
			{Name: "op.segments.[].labels.[].score", Type: "float", Desc: `片段的判定结果可信度`},
			{Name: "op.segments.[].cuts.[].offset", Type: "int", Desc: `截帧的时间位置`},
			{Name: "op.segments.[].cuts.[].result", Type: "object", Desc: `截帧的对应的结果，内容跟具体推理相关`},
		},
	}
)

type Config struct {
	Vframe vframe.VframeParams `json:"default_vframe"`
}

type Set struct {
	Config
	scenario.ServiceSetter
	vod.Service
	vod.OPs
}

var once sync.Once
var set *Set

func Init(s scenario.VideoService, serviceID string) {
	set := GetSet(s, serviceID)
	_ = set
}

func GetSet(s scenario.VideoService, serviceID string) *Set {
	once.Do(func() {
		set = &Set{Config: Config{}}
		set.ServiceSetter = s.Register(
			scenario.ServiceInfo{ID: serviceID, Version: "0.8.0"},
			func(_ops svideo.OPs) error {
				set.OPs = make(vod.OPs)
				for op, of := range _ops {
					set.OPs[op] = of.(svideo.OPFactory)
				}
				return nil
			},
			nil,
		)
		_ = set.GetConfig(context.Background(), &set.Config)
		_ = sync_(set)
	})
	return set
}

func sync_(set *Set) error {

	var (
		uriProxy vframe.URIProxy = vframe.NewURIProxy(
			"http://127.0.0.1:" + set.Router().Port + "/uri")
		once sync.Once
	)

	rs := set.Router().NewRouteSetter("vod",
		func() interface{} {
			once.Do(func() {
				set.Service = vod.NewService(
					context.Background(),
					set.Vframe,
					uriProxy,
					path.Join(set.Workspace(), "data"),
					set.OPs,
				)
			})
			return set.Service
		},
	)
	rs.UpdateRouter(func(
		sf func() middleware.Service,
		newVS func() interface{},
		path func(string) *scenario.Route,
	) error {
		options := []httptransport.ServerOption{
			httptransport.ServerErrorEncoder(func(_ context.Context, err error, w http.ResponseWriter) {
				info, ok := err.(DetectErrorer)
				if ok {
					httpCode, code, desc := info.DetectError()
					transport.ReplyErr(w, httpCode, code, desc)
				} else {
					code, desc := httputil.DetectError(err)
					httputil.ReplyErr(w, code, desc)
				}
			}),
			httptransport.ServerBefore(func(ctx context.Context, req *http.Request) context.Context {
				return xlog.NewContextWithReq(ctx, req)
			}),
		}
		path("/v1/video/{vid}").Doc(_DOC_SYNC_VIDEO).Route().Methods("POST").Handler(httptransport.NewServer(
			func(ctx context.Context, req0 interface{}) (interface{}, error) {
				req := req0.(video.VideoRequest)
				return newVS().(vod.Service).Video(ctx, req)
			},
			func(_ context.Context, r *http.Request) (request interface{}, err error) {
				vars := mux.Vars(r)
				var req video.VideoRequest
				if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
					return nil, err
				}
				req.CmdArgs = []string{vars["vid"]}
				return req, nil
			},
			func(ctx context.Context, w http.ResponseWriter, response interface{}) error {
				w.Header().Set("Content-Type", "application/json; charset=utf-8")
				return json.NewEncoder(w).Encode(response)
			},
			options...,
		))
		return nil
	})
	set.Router().Path("/uri/{uri}").HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		vars := mux.Vars(r)
		uriProxy.GetUri_(context.Background(),
			&struct {
				CmdArgs []string
			}{CmdArgs: []string{vars["uri"]}},
			&restrpc.Env{W: w, Req: r},
		)
	})

	return nil
}
