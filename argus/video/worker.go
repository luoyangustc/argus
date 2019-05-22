package video

import (
	"context"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net/http"
	"net/url"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"gopkg.in/mgo.v2"
	"gopkg.in/mgo.v2/bson"

	"github.com/qiniu/db/mgoutil.v3"
	"github.com/qiniu/http/httputil.v1"
	"github.com/qiniu/rpc.v3"
	xlog "github.com/qiniu/xlog.v1"
)

type Worker interface {
	Run()
	Stop()
}

type WorkerConfig struct {
	MaxPool     int32         `json:"pool_max"`
	TaskTickerS time.Duration `json:"task_ticker_second"`
}

type _Worker struct {
	WorkerConfig

	Jobs
	sync.RWMutex
	runOneJobFunc func(context.Context, Job) (json.RawMessage, error)

	ch            chan bool
	count_running int32
	closed        chan bool
	wg            sync.WaitGroup
	ops           map[string]int32 // TODO init ops when create worker
}

func NewWorkerV2(
	config WorkerConfig, jobs Jobs, ops map[string]int32, saverHook SaverHook,
	runOneJobFunc func(context.Context, Job) (json.RawMessage, error),
) *_Worker {
	xlog.NewWith("NewWorkerV2").Info("op workers:", ops)
	return &_Worker{
		WorkerConfig:  config,
		Jobs:          jobs,
		runOneJobFunc: runOneJobFunc,
		ch:            make(chan bool, config.MaxPool),
		count_running: 0,
		closed:        make(chan bool),
		ops:           ops,
	}
}

func NewWorker(config WorkerConfig,
	jobs Jobs, video Video, ops OPs, saverHook SaverHook,
) *_Worker {
	_ops := make(map[string]int32)
	for name, op := range ops.Load() {
		_ops[name] = op.Count()
	}
	return &_Worker{
		WorkerConfig: config,

		Jobs: jobs,
		runOneJobFunc: func(ctx context.Context, job Job) (json.RawMessage, error) {
			ret, err := runOneVodJob(ctx, video, ops, saverHook, job)
			var msg json.RawMessage
			if ret != nil {
				msg, _ = json.Marshal(ret)
			}
			return msg, err
		},

		ch:            make(chan bool, config.MaxPool),
		count_running: 0,
		closed:        make(chan bool),

		ops: _ops,
	}
}

func NewLiveWorker(config WorkerConfig,
	jobs Jobs, video Video, ops OPs, saverHook SaverHook,
	live LiveService,
) *_Worker {
	_ops := make(map[string]int32)
	for name, op := range ops.Load() {
		_ops[name] = op.Count()
	}

	xlog.NewWith("LiveWoker").Info("op workers:", _ops)
	return &_Worker{
		WorkerConfig: config,

		Jobs: jobs,
		runOneJobFunc: func(ctx context.Context, job Job) (json.RawMessage, error) {
			err := runOneLiveJob(ctx, live, job)
			return nil, err
		},

		ch:            make(chan bool, config.MaxPool),
		count_running: 0,
		closed:        make(chan bool),

		ops: _ops,
	}
}

func (w *_Worker) Stop() {
	close(w.closed)
	w.wg.Wait()
}

func (w *_Worker) Run() {
	var (
		xl     = xlog.NewWith("worker")
		ctx    = xlog.NewContext(context.Background(), xl)
		ticker *time.Ticker
		// ticker = time.NewTicker(time.Second * 10)
	)
	if w.WorkerConfig.TaskTickerS == 0 {
		ticker = time.NewTicker(time.Second * 10)
	} else {
		ticker = time.NewTicker(w.WorkerConfig.TaskTickerS * time.Second)
	}
	w.wg.Add(1)
	defer w.wg.Done()
	for {
		select {
		case <-w.closed:
			xl.Info("run end ...")
			return
		case <-ticker.C:
		case <-w.ch:
		}
		xl.Info("try execute jobs.")

		running := atomic.LoadInt32(&w.count_running)
		if running >= w.MaxPool {
			continue
		}
		ops := make(map[string]int)
		{
			w.RLock()
			for op, _ := range w.ops {
				ops[op] = int(w.ops[op])
			}
			w.RUnlock()
		}

		jobs, _ := w.Jobs.Execute(ctx, int(w.MaxPool-running), ops)
		xl.Infof("execute jobs: %d", len(jobs))

		for _, job := range jobs {

			atomic.AddInt32(&w.count_running, 1)
			{
				w.Lock()
				for _, op := range job.Request.Ops {
					w.ops[op.OP]--
				}
				w.Unlock()
			}
			go func(ctx1 context.Context, job Job) {

				ctx, cancelFunc := context.WithCancel(ctx1)

				defer func() {
					atomic.AddInt32(&w.count_running, -1)
					{
						w.Lock()
						for _, op := range job.Request.Ops {
							w.ops[op.OP]++
						}
						w.Unlock()
					}
					w.ch <- true
				}()

				var (
					ch      = make(chan bool)
					ticker0 = time.NewTicker(time.Second * 10) // TODO config
				)

				go func(ctx context.Context) {
					defer ticker0.Stop()
					for {
						var done = false
						select {
						case <-ticker0.C:
							if w.Jobs.Cancelled(ctx, job) == nil {
								cancelFunc()
							}
							_ = w.Jobs.Touch(ctx, &job)
						case <-ch:
							done = true
							break
						}
						if done {
							break
						}
					}
				}(ctx)

				defer func() {
					close(ch)
				}()

				var err error
				job.Result, err = w.runOneJobFunc(ctx, job)
				if err != nil {
					job.Error = err.Error()
				}
				_ = w.Jobs.Finish(ctx, job)
				xl.Infof("job %s, vid %s done, status: %s, error: %s", job.ID, job.VID, job.Status, job.Error)
			}(context.Background(), job)
		}
		xl.Infof("running jobs: %d, remaining: %d", atomic.LoadInt32(&w.count_running), w.MaxPool-atomic.LoadInt32(&w.count_running))
	}
}

func runOneLiveJob(ctx context.Context, wLive LiveService, job Job) error {
	xl := xlog.FromContextSafe(ctx)
	xl.Infof("Req: %#v", job.Request)
	err := wLive.RunLive(ctx, job.Request, job.ID, OPEnv{
		Uid:   job.UID,
		Utype: job.UType,
	})
	return err
}

func runOneVodJob(ctx context.Context,
	wVideo Video, wOPs OPs, wSaver SaverHook, job Job,
) (map[string]EndResult, error) {

	var (
		xl       = xlog.FromContextSafe(ctx)
		opParams = make(map[string]OPParams)
		result   = struct {
			items map[string]EndResult
			sync.Mutex
		}{items: make(map[string]EndResult)}

		call = func(ctx context.Context, op, url string, result EndResult) error {
			var (
				xl = xlog.FromContextSafe(ctx)
			)
			xl.Infof("try to call: %s %s", op, url)
			err := rpc.DefaultClient.CallWithJson(ctx, nil, "POST", url,
				struct {
					ID      string `json:"id"`
					OP      string `json:"op"`
					Code    int    `json:"code"`
					Message string `json:"message"`
					Result  struct {
						Labels   []ResultLabel   `json:"labels"`
						Segments []SegmentResult `json:"segments"`
					} `json:"result"`
				}{
					ID:      job.VID,
					OP:      op,
					Code:    result.Code,
					Message: result.Message,
					Result:  result.Result,
				})
			xl.Infof("callback done. %s %v %s", op, err, url)
			return nil
		}
		hooks = func(op string) EndHook {
			return EndHookFunc(func(ctx context.Context, rest EndResult) error {
				for _, _op := range job.Request.Ops {
					if _op.OP != op {
						continue
					}
					if _op.HookURL != "" {
						_ = call(ctx, op, _op.HookURL, rest)
					}
				}
				result.Lock()
				result.items[op] = rest
				result.Unlock()
				return nil
			})
		}
		saverOPHook SaverOPHook
		err         error
	)
	for _, op := range job.Request.Ops {
		opParams[op.OP] = op.Params
	}
	ops, ok := wOPs.Create(ctx, opParams, OPEnv{Uid: job.UID, Utype: job.UType})
	if !ok {
		return nil, errors.New("no enough op to work")
	}
	xl.Infof("OPS: %#v %#v", wOPs, ops)

	requestsCounter("video_async", "SB", "", "").Inc()
	requestsParallel("video_async", "S").Inc()
	defer func(begin time.Time) {
		requestsParallel("video_async", "S").Dec()
		responseTimeLong("video_async", "S", "", "").
			Observe(durationAsFloat64(time.Since(begin)))
	}(time.Now())

	if wSaver != nil && job.Request.Params.Save != nil {
		saverOPHook, err = wSaver.Get(ctx, job.UID, job.VID, *job.Request.Params.Save)
		if err != nil {
			xl.Warnf("Save %v", err)
		}
	}
	job.Request.Data.URI, err = func(uri string, uid uint32) (string, error) {
		_url, err := url.Parse(uri)
		if err != nil {
			return uri, err
		}
		if _url.Scheme != "qiniu" {
			return uri, nil
		}
		_url.User = url.User(strconv.Itoa(int(uid)))
		return _url.String(), nil
	}(job.Request.Data.URI, job.UID)
	if err != nil {
		xl.Warnf("improve failed. %s %v", job.Request.Data.URI, err)
		return nil, err
	}

	err = wVideo.Run(ctx, job.Request, ops, saverOPHook, hooks, nil, nil) // TODO
	xl.Infof("%#v %v", result.items, err)

	if url := job.Request.Params.HookURL; url != "" {
		r := struct {
			ID     string               `json:"id"`
			Meta   json.RawMessage      `json:"meta,omitempty"`
			Result map[string]EndResult `json:"result,omitempty"`
		}{
			ID:     job.VID,
			Meta:   job.Request.Data.Attribute.Meta,
			Result: result.items,
		}
		err := rpc.DefaultClient.CallWithJson(ctx, nil, "POST", url, r)
		xl.Infof("callback done. %v %s", err, url)
	}
	return result.items, err
}

////////////////////////////////////////////////////////////////////////////////

type JobStatus string

const (
	StatusWaiting    JobStatus = "WAITING"
	StatusDoing      JobStatus = "DOING"
	StatusFinished   JobStatus = "FINISHED"
	StatusCancelling JobStatus = "Cancelling"
	StatusCancelled  JobStatus = "Cancelled"
)

type Job struct {
	ID  string `json:"id"`
	UID uint32 `json:"-"`

	VID     string          `json:"vid"`
	UType   uint32          `json:"-"`
	Request VideoRequest    `json:"request"`
	Status  JobStatus       `json:"status"`
	Error   string          `json:"error,omitempty"`
	Result  json.RawMessage `json:"result,omitempty"`

	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`

	Ref interface{} `json:"-"`
	OPs []string    `json:"-"`
}

type Jobs interface {
	Submit(context.Context, uint32, uint32, string, VideoRequest) (string, error)

	Execute(context.Context, int, map[string]int) ([]Job, error)
	Touch(context.Context, *Job) error
	Finish(context.Context, Job) error
	Cancel(context.Context, uint32, string) error
	Cancelled(context.Context, Job) error

	Get(context.Context, uint32, string) (Job, error)
	List(context.Context, *uint32, *JobStatus, *time.Time, *time.Time, *string, *int) ([]Job, string, error)
}

//----------------------------------------------------------------------------//

type JobsInMgoConfig struct {
	IdleJobTimeout time.Duration  `json:"idle_job_timeout"`
	MgoPoolLimit   int            `json:"mgo_pool_limit"`
	Mgo            mgoutil.Config `json:"mgo"`
}

type _JobInMgo struct {
	ID  bson.ObjectId `bson:"_id,omitempty"`
	UID uint32        `bson:"uid"`

	VID        string               `bson:"vid"`
	UType      uint32               `bson:"utype"`
	Request    VideoRequest         `bson:"request"`
	Status     JobStatus            `bson:"status"`
	Error      string               `bson:"error"`
	Result     map[string]EndResult `bson:"result,omitempty"` // 保留向前兼容
	ResultJson json.RawMessage      `bson:"result_json,omitempty"`
	OPs        string               `bson:"ops,omitempty"`

	CreatedAt time.Time `bson:"created_at"`
	UpdatedAt time.Time `bson:"updated_at"`

	Ver int `bson:"_ver"`
}

var _ Jobs = _JobsInMgo{}

func (job _JobInMgo) To() Job {
	return Job{
		ID:      job.ID.Hex(),
		UID:     job.UID,
		VID:     job.VID,
		UType:   job.UType,
		Request: job.Request,
		Status:  job.Status,
		Error:   job.Error,
		Result: func() json.RawMessage {
			if job.ResultJson != nil {
				return job.ResultJson
			}
			if job.Result != nil {
				var msg json.RawMessage
				msg, _ = json.Marshal(job.Result)
				return msg
			}
			return nil
		}(),
		CreatedAt: job.CreatedAt,
		UpdatedAt: job.UpdatedAt,
		Ref:       job,
		OPs:       decodeOPs(job.OPs),
	}
}

func (job _JobInMgo) Update(_job *Job) {
	_job.ID = job.ID.Hex()
	_job.UID = job.UID
	_job.VID = job.VID
	_job.UType = job.UType
	_job.Request = job.Request
	_job.Status = job.Status
	_job.Error = job.Error
	if job.ResultJson != nil {
		_job.Result = job.ResultJson
	} else if job.Result != nil {
		_job.Result, _ = json.Marshal(job.Result)
	} else {
		_job.Result = nil
	}
	_job.CreatedAt = job.CreatedAt
	_job.UpdatedAt = job.UpdatedAt
	_job.Ref = job
	_job.OPs = decodeOPs(job.OPs)
}

type _JobsInMgo struct {
	JobsInMgoConfig

	coll mgoutil.Collection
}

func NewJobsInMgo(conf JobsInMgoConfig) (_JobsInMgo, error) {

	var (
		mgoSessionPoolLimit = 100
		colls               = struct {
			Jobs mgoutil.Collection `coll:"jobs"`
		}{}
	)
	sess, err := mgoutil.Open(&colls, &conf.Mgo)
	if err != nil {
		return _JobsInMgo{}, err
	}
	if conf.MgoPoolLimit > 0 {
		mgoSessionPoolLimit = conf.MgoPoolLimit
	}
	sess.SetPoolLimit(mgoSessionPoolLimit)

	colls.Jobs.EnsureIndex(mgo.Index{Key: []string{"uid"}})
	colls.Jobs.EnsureIndex(mgo.Index{Key: []string{"status"}})
	colls.Jobs.EnsureIndex(mgo.Index{Key: []string{"created_at"}})
	colls.Jobs.EnsureIndex(mgo.Index{Key: []string{"updated_at"}})

	return _JobsInMgo{JobsInMgoConfig: conf, coll: colls.Jobs}, nil
}

func (m _JobsInMgo) Submit(
	ctx context.Context, uid, utype uint32, vid string, req VideoRequest,
) (string, error) {
	var (
		job = _JobInMgo{
			ID:        bson.NewObjectId(),
			UID:       uid,
			VID:       vid,
			UType:     utype,
			Request:   req,
			Status:    StatusWaiting,
			CreatedAt: time.Now(),
			UpdatedAt: time.Now(),
			Ver:       0,
		}
		ops []string
	)
	for _, op := range req.Ops {
		ops = append(ops, op.OP)
	}
	job.OPs = encodeOPs(ops)

	coll := m.coll.CopySession()
	defer coll.CloseSession()

	return job.ID.Hex(), coll.Insert(job)
}

func (m _JobsInMgo) update(
	ctx context.Context,
	coll mgoutil.Collection,
	job *_JobInMgo,
) error {
	job.Ver = job.Ver + 1
	job.UpdatedAt = time.Now()
	// TODO 细分错误
	return coll.Update(
		bson.M{"_id": job.ID, "_ver": job.Ver - 1},
		job,
	)
}

func genOPs(ops map[string]int) []string {
	var (
		ret = []string{}
		src []string
	)
	for op, count := range ops {
		if count > 0 {
			src = append(src, op)
		}
	}
	sort.Strings(src)
	length := len(src)

	for i := 1; i < int(math.Pow(2.0, float64(length))); i++ {
		var (
			index = 0
			j     = i
			temp  []string
		)
		for j > 0 {
			if j%2 == 1 {
				temp = append(temp, src[index])
			}
			index++
			j = j / 2
		}
		ret = append(ret, encodeOPs(temp))
	}
	return ret
}

func encodeOPs(ops []string) string {
	sort.Strings(ops)
	return strings.Join(ops, "|")
}

func decodeOPs(str string) []string {
	if len(str) == 0 {
		return nil
	}
	return strings.Split(str, "|")
}

func (m _JobsInMgo) Execute(ctx context.Context, limit int, ops map[string]int) ([]Job, error) {
	coll := m.coll.CopySession()
	defer coll.CloseSession()

	var (
		jobs = make([]Job, 0, limit)
		now  = time.Now()
	)

	for {
		var (
			_len  = len(jobs)
			_jobs = make([]_JobInMgo, 0)
		)
		if _len >= limit {
			break
		}
		xlog.FromContextSafe(ctx).Debugf("find: %v", genOPs(ops))
		if m.IdleJobTimeout > 0 {
			err := coll.
				Find(
					bson.M{
						"status":     StatusDoing,
						"updated_at": bson.M{"$lt": now.Add(-1 * m.IdleJobTimeout)},
						"$or":        []bson.M{bson.M{"ops": bson.M{"$in": genOPs(ops)}}, bson.M{"ops": bson.M{"$exists": false}}},
					}).
				Limit(limit - _len).
				All(&_jobs)
			if err != nil {
				return jobs, err
			}
		}
		if len(_jobs) == 0 {
			err := coll.
				Find(bson.M{
					"status": StatusWaiting,
					"$or":    []bson.M{bson.M{"ops": bson.M{"$in": genOPs(ops)}}, bson.M{"ops": bson.M{"$exists": false}}},
				}).
				Limit(limit - _len).
				All(&_jobs)
			if err != nil {
				return jobs, err
			}
		}
		if len(_jobs) == 0 {
			break
		}
	Loop:
		for _, _job := range _jobs {
			for _, op := range decodeOPs(_job.OPs) {
				if ops[op] <= 0 {
					continue Loop
				}
			}
			for _, op := range decodeOPs(_job.OPs) {
				ops[op]--
			}
			_job.Status = StatusDoing
			err := m.update(ctx, coll, &_job)
			if err == mgo.ErrNotFound {
				continue
			}
			if err != nil {
				return jobs, err
			}
			jobs = append(jobs, _job.To())
		}
	}

	return jobs, nil
}

func (m _JobsInMgo) Touch(ctx context.Context, job *Job) error {
	coll := m.coll.CopySession()
	defer coll.CloseSession()

	var (
		_job = job.Ref.(_JobInMgo)
	)
	err := m.update(ctx, coll, &_job)
	if err != nil {
		return err
	}
	_job.Update(job)
	return nil
}

func (m _JobsInMgo) Cancelled(ctx context.Context, job Job) error {
	coll := m.coll.CopySession()
	defer coll.CloseSession()

	var (
		_job = _JobInMgo{}
	)
	d, err := hex.DecodeString(job.ID)
	if err != nil || len(d) != 12 {
		return httputil.NewError(http.StatusNotFound, fmt.Sprintf("invalid input Job id: %q", job.ID))
	}
	if err = coll.Find(bson.M{"_id": bson.ObjectId(d), "uid": job.UID, "status": StatusCancelling}).One(&_job); err != nil {
		return err
	}
	_job.Status = StatusCancelled

	if err = m.update(ctx, coll, &_job); err != nil {
		return err
	}
	return nil
}

func (m _JobsInMgo) Cancel(ctx context.Context, uid uint32, id string) error {
	coll := m.coll.CopySession()
	defer coll.CloseSession()

	var (
		_job = _JobInMgo{}
	)

	d, err := hex.DecodeString(id)
	if err != nil || len(d) != 12 {
		return httputil.NewError(http.StatusNotFound, fmt.Sprintf("invalid input Job id: %q", id))
	}

	if err = coll.Find(bson.M{"_id": bson.ObjectId(d), "uid": uid, "status": bson.M{"$in": []JobStatus{StatusWaiting, StatusDoing}}}).One(&_job); err != nil {
		return err
	}

	_job.Status = StatusCancelling
	if err = m.update(ctx, coll, &_job); err != nil {
		return err
	}

	return nil
}

func (m _JobsInMgo) Finish(ctx context.Context, job Job) error {
	coll := m.coll.CopySession()
	defer coll.CloseSession()

	var (
		_job = job.Ref.(_JobInMgo)
	)
	_job.Status = StatusFinished
	_job.Error = job.Error
	_job.ResultJson = job.Result
	err := m.update(ctx, coll, &_job)
	if err != nil {
		return err
	}
	return nil
}

func (m _JobsInMgo) Get(ctx context.Context, uid uint32, id string) (Job, error) {
	coll := m.coll.CopySession()
	defer coll.CloseSession()

	var (
		_job = _JobInMgo{}
	)

	d, err := hex.DecodeString(id)
	if err != nil || len(d) != 12 {
		return Job{}, httputil.NewError(http.StatusNotFound, fmt.Sprintf("invalid input Job id: %q", id))
	}

	err = coll.Find(bson.M{"_id": bson.ObjectId(d), "uid": uid}).One(&_job)
	if err != nil {
		return Job{}, err
	}
	return _job.To(), nil
}

func (m _JobsInMgo) List(
	ctx context.Context,
	uid *uint32,
	status *JobStatus,
	created_from, created_to *time.Time,
	marker *string,
	limit *int,
) ([]Job, string, error) {
	coll := m.coll.CopySession()
	defer coll.CloseSession()

	var (
		_jobs         = make([]_JobInMgo, 0)
		_query        = bson.M{}
		_query_create = bson.M{}
		nextMarker    string
	)
	if uid != nil {
		_query["uid"] = *uid
	}
	if status != nil && *status != "" {
		_query["status"] = *status
	}
	// create_from <= _job.create_at < create_to
	if created_from != nil {
		_query_create["$gte"] = created_from
	}
	if created_to != nil {
		_query_create["$lt"] = created_to
	}
	if len(_query_create) > 0 {
		_query["created_at"] = _query_create
	}

	if marker != nil && *marker != "" {
		if !bson.IsObjectIdHex(*marker) {
			return nil, "", errors.New("invalid marker")
		}
		_query["_id"] = bson.M{"$gt": bson.ObjectIdHex(*marker)}
	}

	query := coll.Find(_query).Sort("created_at")

	// 获取limit+1条，以确定是否还有后续documents
	if limit != nil && *limit > 0 {
		query.Limit(*limit + 1)
	}

	err := query.All(&_jobs)
	if err != nil {
		return nil, "", err
	}
	if limit != nil && *limit > 0 && len(_jobs) > *limit {
		nextMarker = _jobs[*limit-1].ID.Hex()
		_jobs = _jobs[:*limit]
	}
	var jobs = make([]Job, 0, len(_jobs))
	for _, _job := range _jobs {
		jobs = append(jobs, _job.To())
	}
	return jobs, nextMarker, nil
}
