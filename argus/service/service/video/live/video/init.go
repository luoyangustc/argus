package video

import (
	"context"
	"encoding/json"
	"net/http"
	"path"
	"strconv"
	"sync"
	"time"

	httptransport "github.com/go-kit/kit/transport/http"
	"github.com/gorilla/mux"
	"github.com/imdario/mergo"

	mgoutil "github.com/qiniu/db/mgoutil.v3"
	httputil "github.com/qiniu/http/httputil.v1"
	xlog "github.com/qiniu/xlog.v1"

	"qiniu.com/argus/service/middleware"
	scenario "qiniu.com/argus/service/scenario/video"
	. "qiniu.com/argus/service/service"
	svideo "qiniu.com/argus/service/service/video"
	"qiniu.com/argus/service/service/video/live"
	"qiniu.com/argus/service/transport"
	"qiniu.com/argus/video"
	video0 "qiniu.com/argus/video"
	"qiniu.com/argus/video/vframe"
)

const (
	ASYNC_VERSION = "1.0.0"
	ID            = "qiniu.com/argus/service/service/video/live/video"
)

type Config struct {
	Vframe      vframe.VframeParams    `json:"default_vframe"`
	SaverConfig *svideo.FileSaveConfig `json:"default_save"`
	Jobs        video0.JobsInMgoConfig `json:"jobs"`
	Worker      struct {
		MaxPool     int32         `json:"pool_max"`
		TaskTickerS time.Duration `json:"task_ticker_second"`
	} `json:"worker"`
}

func (c *Config) UnmarshalJSON(b []byte) error {
	type C Config
	c2 := C{}
	if err := json.Unmarshal(b, &c2); err != nil {
		return err
	}
	c1 := C(_DEFAULT_ASYNC_CONFIG)
	_ = mergo.Merge(&c2, &c1)
	*c = Config(c2)
	return nil
}

var (
	_DEFAULT_ASYNC_CONFIG = Config{
		Vframe: vframe.VframeParams{
			Mode: func() *int {
				mode := 2
				return &mode
			}(),
			Interval: 25,
		},
		Jobs: video0.JobsInMgoConfig{
			IdleJobTimeout: 30000000000,
			MgoPoolLimit:   20,
			Mgo: mgoutil.Config{
				Host:           "127.0.0.1:27017",
				DB:             "argus_live",
				Mode:           "strong",
				SyncTimeoutInS: 5,
			},
		},
		Worker: struct {
			MaxPool     int32         `json:"pool_max"`
			TaskTickerS time.Duration `json:"task_ticker_second"`
		}{
			MaxPool:     10,
			TaskTickerS: 10,
		},
	}
	_DOC_CREATE = scenario.APIDoc{
		Name:    `新建异步视频任务`,
		Version: ASYNC_VERSION,
		Desc:    []string{`新建异步视频分析任务，指定分析内容和相关参数`},
		Request: `POST /v1/video/<vid>/async HTTP/1.1
	Context-Type: application/json

	{
		"data": {
			"uri": "rtsp://xx"
		},
		"params": {
			"vframe": {
				"interval": <interval:float>
			},
			"live": {
				"timeout": <timeout:float>,
				"downstream": "rtsp://xx.com/yyy"
			},
			"hookURL": "http://yy.com/yyy",
			"save": {
				"prefix": <save_path_prefix>
			}
		},
		"ops": [
			{
				"op": <op:string>,
				"cut_hook_url": "http://yy.com/yyy",
				"params": {
					...
				}
			},
			...
		]
	}`,
		Response: `200 OK
	Content-Type: application/json

	{
		"jod": <job_id:string>
	}`,
		RequestParam: []scenario.APIDocParam{
			{Name: "vid", Type: "string", Must: "Yes", Desc: "视频唯一标识，异步处理的返回结果中会带上该信息"},
			{Name: "data.uri", Type: "string", Must: "Yes", Desc: "视频流地址"},
			{Name: "params.vframe.interval", Type: "int", Must: "No", Desc: "跳帧间隔，每间隔interval帧推理一帧，interval=1表示全帧推理"},
			{Name: "params.live.timeout", Type: "float", Must: "No", Desc: "视频流超时判定参数，单位sec（秒），仅直播有效，默认30秒"},
			{Name: "params.live.downstream", Type: "string", Must: "No", Desc: "下行推流地址，该字段为空表示无下行流，仅直播有效,当前仅支持rtsp、rtmp协议"},
			{Name: "params.hookURL", Type: "string", Must: "No", Desc: "视频分析结束后的回调地址"},
			{Name: "params.save", Type: `{}struc`, Must: "No", Desc: "是否开启截帧保存，保存对应推理图片"},
			{Name: "params.save.prefix", Type: "string", Must: "No", Desc: "推理图片保存路径前缀"},
			{Name: "ops.op", Type: "string", Must: "Yes", Desc: "执行的推理cmd"},
			{Name: "ops.op.cut_hook_url", Type: "string", Must: "No", Desc: "帧推理结果回调地址"},
			{Name: "ops.op.params", Type: `{}struc`, Must: "No", Desc: "推理命令参数，跟具体推理cmd有关"},
		},
		ResponseParam: []scenario.APIDocParam{
			{Name: "job_id", Type: "string", Desc: "视频异步任务唯一标识"},
		},
	}

	_DOC_KILL = scenario.APIDoc{
		Name:     "停止异步任务",
		Version:  ASYNC_VERSION,
		Desc:     []string{`结束指定的异步视频分析任务`},
		Request:  `POST /v1/jobs/<job_id>/kill HTTP/1.1`,
		Response: `200 OK`,
		RequestParam: []scenario.APIDocParam{
			{Name: "job_id", Type: "string", Must: "Yes", Desc: "视频异步任务任务的唯一标识，新建异步任务时返回的结果"},
		},
	}
	_DOC_JOBS = scenario.APIDoc{
		Name:    `查询所有异步任务`,
		Version: ASYNC_VERSION,
		Desc:    []string{`查询所有异步任务`},
		Request: `GET /v1/jobs?status=<job_status>&created_from=<created_from>&created_to=<created_to>&marker=<marker>&limit=<limit> HTTP/1.1`,
		Response: `200 OK
	Content-Type: application/json

	{
		"jobs": [
			{
				"id": <job_id>,
				"vid": <video_id>,
				"status": <job_status>,
				"created_at": <create_timestamp>
				"updated_at": <updapte_timestamp>
			}
			...
		],
		"marker":<Marker>
	}`,
		RequestParam: []scenario.APIDocParam{
			{Name: "status", Type: "string", Must: "No", Desc: `指定查询任务状态["WAITING","DOING","FINISHED","Cancelling","Cancelled"],默认为空，即查询所有任务`},
			{Name: "created_from", Type: "int", Must: "No", Desc: `视频异步任务查询的创建时间区间起点，格式为unix时间戳,时间区间：[created_from,created_to)`},
			{Name: "created_to	", Type: "int", Must: "No", Desc: `视频异步任务查询的创建时间区间终点，格式为unix时间戳,时间区间：[created_from,created_to)`},
			{Name: "marker	", Type: "string", Must: "No", Desc: `上一次列举返回的位置标记，作为本次列举的起点信息。默认值为空字符串`},
			{Name: "limit	", Type: "int", Must: "No", Desc: `本次列举的条目数，范围为 1-1000。默认值为 1000`},
		},
		ResponseParam: []scenario.APIDocParam{
			{Name: "jobs.id", Type: "string", Desc: "视频异步任务唯一标识"},
			{Name: "jobs.vid", Type: "string", Desc: "视频一标识，异步处理的返回结果中会带上该信息"},
			{Name: "jobs.status", Type: "string", Desc: "视频异步任务状态"},
			{Name: "jobs.created_at", Type: "string", Desc: "视频异步任务创建时间"},
			{Name: "jobs.updated_at", Type: "string", Desc: "视频异步任务最近更新时间"},
			{Name: "marker", Type: "string", Desc: "有剩余条目则返回非空字符串，作为下一次列举的参数传入。如果没有剩余条目则返回空字符串"},
		},
	}

	_DOC_JOB_DETAIL = scenario.APIDoc{
		Name:    `查询异步任务详情`,
		Version: ASYNC_VERSION,
		Desc:    []string{`根据job_id查询异步任务详情`},
		Request: `GET /v1/jobs/<job_id> HTTP/1.1`,
		Response: `200 OK
	Content-Type: application/json

	{
		"id": <job_id>
		"vid": <video_id>,
		"request": {

			"data": {
				"uri": "rtsp://xx"
			},
			"params": {
				"vframe": {
					"interval": <interval:float>
				},
				"live": {
					"timeout": <timeout:float>
				},
				"hookURL": "http://yy.com/yyy",
				"save": {
					"prefix": <save_path_prefix>
				}
			},
			"ops": [
				{
					"op": <op:string>,
					"cut_hook_url": "http://yy.com/yyy",
					"params": {
						...
					}
				},
				...
			]
		},
		"status": <job_status>,
		"created_at": <create_timestamp>,
		"updated_at": <updapte_timestamp>,
		"error": <error_message>
	}`,
		RequestParam: []scenario.APIDocParam{
			{Name: "job_id", Type: "string", Must: "Yes", Desc: "异步任务的唯一标识，新建异步任务时返回的结果"},
		},
		ResponseParam: []scenario.APIDocParam{
			{Name: "id", Type: "string", Desc: "视频异步任务唯一标识"},
			{Name: "vid", Type: "string", Desc: "视频流唯一标识，异步处理的返回结果中会带上该信息"},
			{Name: "request.data.uri", Type: "string", Desc: "视频流地址"},
			{Name: "request.data.uri", Type: "string", Desc: "视频流地址"},
			{Name: "request.params.vframe.interval", Type: "int", Desc: "跳帧间隔，每间隔interval帧推理一帧，interval=1表示全帧推理"},
			{Name: "request.params.live.timeout", Type: "float", Desc: "视频流超时判定参数，单位sec（秒）,仅对直播有效"},
			{Name: "request.params.live.downstream", Type: "string", Desc: "下行推流地址，该字段为空表示无下行流，仅对直播有效"},
			{Name: "request.params.hookURL", Type: "string", Desc: "视频分析结束后的回调地址"},
			{Name: "request.params.save", Type: `{}struc`, Desc: "是否开启截帧保存，保存对应推理图片"},
			{Name: "request.params.save.prefix", Type: "string", Desc: "推理图片保存路径前缀"},
			{Name: "request.ops.op", Type: "string", Desc: "执行的推理cmd"},
			{Name: "request.ops.op.cut_hook_url", Type: "string", Desc: "推理结果回调地址"},
			{Name: "request.ops.op.params", Type: `{}struc`, Desc: "推理命令参数，跟具体推理cmd有关"},
			{Name: "status", Type: "string", Desc: "视频异步任务状态"},
			{Name: "created_at", Type: "string", Desc: "视频异步任务创建时间"},
			{Name: "updated_at", Type: "string", Desc: "视频异步任务最近更新时间"},
			{Name: "error", Type: "string", Desc: "处理视频流的过程中遇到的错误，会返回相应的错误信息"},
		},
	}
	_DOC_FRAME_CALLBACK = scenario.APIDoc{
		Name:    `帧回调`,
		Version: ASYNC_VERSION,
		Desc:    []string{`帧推理结果HTTP回调`},
		Request: `POST /xxxxxxx HTTP/1.1
	Content-Type: application/json

	{
		"vid": <live_id:string>,
		"job_id": <job_id:string>
		"op": <op:string>,
		"offset": <offset:int>,
		"uri": <uri:string>,
		"result": {}
	}`,
		RequestParam: []scenario.APIDocParam{
			{Name: "vid", Type: "string", Must: "Yes", Desc: `异步任务唯一标识`},
			{Name: "live_id", Type: "string", Must: "Yes", Desc: `视频唯一标识，申请任务时传入的video_id`},
			{Name: "op", Type: "string", Must: "Yes", Desc: `推理cmd`},
			{Name: "offset", Type: "int", Must: "Yes", Desc: `截帧时间戳`},
			{Name: "uri", Type: "string", Must: "No", Desc: `截帧缓存路径，申请任务时需开启save`},
			{Name: "result", Type: "object", Must: "Yes", Desc: `截帧的对应的结果，内容跟具体推理相关`},
		},
	}
	_DOC_JOB_CALLBACK = scenario.APIDoc{
		Name:    `任务结果回调`,
		Version: ASYNC_VERSION,
		Desc:    []string{`视频流推理结果HTTP回调`},
		Request: `POST /xxxxxxx HTTP/1.1
	Content-Type: application/json

	{
		"vid": <id:string>,
		"job_id": <job_id:string>,
		"error": <error:string>,
		"result": <result:object>
	}`,
		RequestParam: []scenario.APIDocParam{
			{Name: "vid", Type: "string", Must: "Yes", Desc: `视频唯一标识，申请任务时传入的video_id`},
			{Name: "job_id", Type: "string", Must: "Yes", Desc: `异步任务唯一标识`},
			{Name: "error", Type: "string", Must: "No", Desc: `异步任务错误详细信息,正确处理时为空`},
			{Name: "result", Type: "object", Must: "No", Desc: `异步任务结果,直播任务为空`},
		},
	}
)

type Set struct {
	Config
	scenario.ServiceSetter
	live.Service
	live.OPs
}

var once sync.Once
var set *Set

func Init(s scenario.VideoService, serviceID string) {
	set := GetSet(s, serviceID)
	_ = set
}

func GetSet(s scenario.VideoService, serviceID string) *Set {
	once.Do(func() {
		set = &Set{Config: _DEFAULT_ASYNC_CONFIG}
		set.ServiceSetter = s.Register(
			scenario.ServiceInfo{ID: serviceID, Version: "0.8.0"},
			func(_ops svideo.OPs) error {
				set.OPs = make(live.OPs)
				for op, of := range _ops {
					set.OPs[op] = of.(svideo.OPFactory)
				}
				return nil
			},
			func() {
				var jobs video.Jobs
				var err error
				if jobs, err = video.NewJobsInMgo(set.Config.Jobs); err != nil {
					panic("async: fail to create jobs, error:" + err.Error())
				}
				set.Service = live.NewService(
					context.Background(),
					live.Config{
						DefaultVframeParams: set.Config.Vframe,
						Workerspace:         path.Join(set.Workspace(), "data"),
						SaverHook: func() svideo.SaverHook {
							if set.Config.SaverConfig == nil {
								return nil
							}
							return svideo.NewFileSaver(*set.Config.SaverConfig)
						}(),
					},
					set.OPs,
					jobs,
				)
				_ops := make(map[string]int32)
				for name, op := range set.OPs {
					_ops[name] = op.Count()
				}
				worker := video0.NewWorkerV2(set.Config.Worker, jobs, _ops, nil,
					func(ctx context.Context, job video0.Job) (msg json.RawMessage, err error) {
						err = set.Live(ctx, job.Request, job.ID)
						return
					})
				go worker.Run()
			},
		)
		_ = set.GetConfig(context.Background(), &set.Config)
		_ = async(set)
	})
	return set
}

func async(set *Set) error {

	gen := func() {}

	var once sync.Once
	// init routers
	rs := set.Router().NewRouteSetter("live", func() interface{} {
		once.Do(gen)
		return set.Service
	})
	rs.UpdateRouter(func(
		sf func() middleware.Service,
		newVS func() interface{},
		path func(string) *scenario.Route,
	) error {

		options := []httptransport.ServerOption{
			// httptransport.ServerErrorLogger(logger),
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
		path("/v1/video/{vid}/async").Doc(_DOC_CREATE).Route().Methods("POST").Handler(httptransport.NewServer(
			func(ctx context.Context, req0 interface{}) (interface{}, error) {
				req := req0.(video.VideoRequest)
				return newVS().(live.Service).Async(ctx, req)
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

		path("/v1/jobs/{jod_id}/kill").Doc(_DOC_KILL).Route().Methods("POST").Handler(httptransport.NewServer(
			func(ctx context.Context, req0 interface{}) (interface{}, error) {
				req := req0.(string)
				return newVS().(live.Service).Kill(ctx, req)
			},
			func(_ context.Context, r *http.Request) (request interface{}, err error) {
				return mux.Vars(r)["jod_id"], nil
			},
			func(ctx context.Context, w http.ResponseWriter, response interface{}) error {
				w.Header().Set("Content-Type", "application/json; charset=utf-8")
				return json.NewEncoder(w).Encode(response)
			},
			options...,
		))
		path("/v1/jobs").Doc(_DOC_JOBS).Route().Methods("GET").Handler(httptransport.NewServer(
			func(ctx context.Context, req0 interface{}) (interface{}, error) {
				req := req0.(live.GetJobsRequest)
				return newVS().(live.Service).GetJobs(ctx, req)
			},
			func(ctx context.Context, r *http.Request) (request interface{}, err error) {
				var req live.GetJobsRequest
				vars := r.URL.Query()
				if len(vars["status"]) == 1 {
					req.Status = vars["status"][0]
				}
				if len(vars["created_from"]) == 1 {
					if req.CreatedFrom, err = strconv.ParseInt(vars["created_from"][0], 10, 64); err != nil {
						return nil, err
					}
				}
				if len(vars["created_to"]) == 1 {
					if req.CreatedTo, err = strconv.ParseInt(vars["created_to"][0], 10, 64); err != nil {
						return nil, err
					}
				}
				if len(vars["marker"]) == 1 {
					req.Marker = vars["marker"][0]
				}
				if len(vars["limit"]) == 1 {
					if req.Limit, err = strconv.Atoi(vars["limit"][0]); err != nil {
						return nil, err
					}
				}
				return req, nil
			},
			func(ctx context.Context, w http.ResponseWriter, response interface{}) error {
				w.Header().Set("Content-Type", "application/json; charset=utf-8")
				return json.NewEncoder(w).Encode(response)
			},
			options...,
		))
		path("/v1/jobs/{job_id}").Doc(_DOC_JOB_DETAIL).Route().Methods("GET").Handler(httptransport.NewServer(
			func(ctx context.Context, req interface{}) (interface{}, error) {
				return newVS().(live.Service).GetJobByID(ctx, req.(string))
			},
			func(_ context.Context, r *http.Request) (request interface{}, err error) {
				return mux.Vars(r)["job_id"], nil
			},
			func(ctx context.Context, w http.ResponseWriter, response interface{}) error {
				w.Header().Set("Content-Type", "application/json; charset=utf-8")
				return json.NewEncoder(w).Encode(response)
			},
			options...,
		))
		return nil
	})
	{
		// 回调文档
		_ = rs.Callback().Doc(_DOC_FRAME_CALLBACK)
		_ = rs.Callback().Doc(_DOC_JOB_CALLBACK)
	}
	return nil
}
