package eval

import (
	"context"
	"encoding/json"
	"net/http"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"github.com/qiniu/http/httputil.v1"
	"github.com/qiniu/rpc.v3"
	"github.com/qiniu/xlog.v1"

	. "qiniu.com/argus/atserving/model"
	"qiniu.com/argus/com/util"
	STS "qiniu.com/argus/sts/client"
)

//----------------------------------------------------------------------------//

// ResponseCaller ...
type ResponseCaller interface {
	SetHeader(error)
	Header() http.Header
	Call(context.Context, interface{}) error
}

var _ ResponseCaller = &responseCaller{}

type responseCaller struct {
	header http.Header

	url string
	id  string
	err error

	*AuditLog
}

func newResponseCaller(url, id string, audit *AuditLog) *responseCaller {
	return &responseCaller{header: make(http.Header), url: url, id: id, AuditLog: audit}
}

func (c *responseCaller) SetHeader(err error) { c.err = err }
func (c *responseCaller) Header() http.Header { return c.header }

func (c *responseCaller) parseError(err error) (int, string) {
	if err == nil {
		return http.StatusOK, ""
	}
	switch err {
	case ErrBadRequest, ErrOutOfBatchSize, ErrInvalidImage:
		return http.StatusBadRequest, err.Error()
	case ErrForwardInference:
		return http.StatusInternalServerError, err.Error()
	default:
		return httputil.DetectError(err)
	}
}

func (c *responseCaller) genMessage(resp interface{}) ResponseMessage {
	var (
		code, text = c.parseError(c.err)
		str, _     = json.Marshal(resp)
	)
	if c.AuditLog != nil {
		c.AuditLog.StatusCode = code
	}
	return ResponseMessage{
		ID:         c.id,
		StatusCode: code,
		StatusText: text,
		Header:     c.header,
		Response:   string(str),
	}
}

func (c *responseCaller) Call(ctx context.Context, resp interface{}) error {
	var (
		xl  = xlog.FromContextSafe(ctx)
		msg = c.genMessage(resp)
		url = c.url + c.id
	)
	c.header[KEY_LOG] = xl.Xget()
	if c.AuditLog != nil {
		c.AuditLog.RespHeader = c.header
		c.AuditLog.RespBody = msg
	}
	err := rpc.DefaultClient.CallWithJson(ctx, nil, "POST", url, msg)
	xl.Infof("call: %s %#v %v", url, msg, err)
	return err
}

//----------------------------------------------------------------------------//

// TaskRequest ...
type TaskRequest interface {
	BatchSize() int
	Request() interface{}
	Reset(interface{})
}

func newTaskRequest(req interface{}) TaskRequest {
	switch v := req.(type) {
	case EvalRequestInner:
		return &_EvalRequest{v}
	case GroupEvalRequestInner:
		return &_GroupEvalRequest{v}
	case []EvalRequestInner:
		return _BatchEvalRequest(v)
	case []GroupEvalRequestInner:
		return _BatchGroupEvalRequest(v)
	default:
		panic("never happen")
	}
}

type _EvalRequest struct {
	EvalRequestInner
}

func (r _EvalRequest) BatchSize() int        { return 1 }
func (r *_EvalRequest) Request() interface{} { return r.EvalRequestInner }
func (r *_EvalRequest) Reset(req interface{}) {
	if v, ok := req.(*_EvalRequest); ok {
		r.EvalRequestInner = v.EvalRequestInner
	}
}

type _GroupEvalRequest struct {
	GroupEvalRequestInner
}

func (r _GroupEvalRequest) BatchSize() int        { return 1 }
func (r *_GroupEvalRequest) Request() interface{} { return r.GroupEvalRequestInner }
func (r *_GroupEvalRequest) Reset(req interface{}) {
	if v, ok := req.(*_GroupEvalRequest); ok {
		r.GroupEvalRequestInner = v.GroupEvalRequestInner
	}
}

type _BatchEvalRequest []EvalRequestInner

func (r _BatchEvalRequest) BatchSize() int       { return len(r) }
func (r _BatchEvalRequest) Request() interface{} { return []EvalRequestInner(r) }
func (r _BatchEvalRequest) Reset(req interface{}) {
	if v, ok := req.(_BatchEvalRequest); ok {
		for i, req2 := range v {
			r[i] = req2
		}
	}
}

type _BatchGroupEvalRequest []GroupEvalRequestInner

func (r _BatchGroupEvalRequest) BatchSize() int       { return len(r) }
func (r _BatchGroupEvalRequest) Request() interface{} { return []GroupEvalRequestInner(r) }
func (r _BatchGroupEvalRequest) Reset(req interface{}) {
	if v, ok := req.(_BatchGroupEvalRequest); ok {
		for i, req2 := range v {
			r[i] = req2
		}
	}
}

//----------------------------------------------------------------------------//

// TaskResult ...
type TaskResult struct {
	IsBatch   bool
	Responses []EvalResponseInner
	err       error
}

// Task ...
type Task interface {
	ID() int64
	Ready()
	IsReady() bool
	BatchSize() int
	Request() interface{}
	Done(TaskResult)
	Watch() <-chan TaskResult
	Close()
}

var _ Task = &task{}

type task struct {
	TaskRequest
	id    int64
	ready uint32
	ch    chan TaskResult
}

// NewTask ...
func NewTask(id int64, req TaskRequest) Task {
	return &task{
		TaskRequest: req,
		id:          id,
		ready:       0,
		ch:          make(chan TaskResult, 1),
	}
}

func (t *task) ID() int64            { return t.id }
func (t *task) Ready()               { atomic.StoreUint32(&t.ready, 1) }
func (t *task) IsReady() bool        { return atomic.LoadUint32(&t.ready) != 0 }
func (t *task) BatchSize() int       { return t.TaskRequest.BatchSize() }
func (t *task) Request() interface{} { return t.TaskRequest.Request() }
func (t *task) Done(result TaskResult) {
	defer func() {
		if err := recover(); err != nil {
			xlog.NewDummy().Warnf("%#v", err)
		}
	}()
	t.ch <- result
}
func (t *task) Watch() <-chan TaskResult { return t.ch }
func (t *task) Close()                   { close(t.ch) }

//----------------------------------------------------------------------------//

// Worker ...
type Worker interface {
	Do(context.Context, TaskRequest, ResponseCaller) (bool, error)
	Run(context.Context, Task)
}

// WorkerConfig ...
type WorkerConfig struct {
	MaxConcurrent int64 `json:"max_concurrent"`
	BatchSize     int64 `json:"batch_size"`
	Wait          int64 `json:"wait"`
}

var _ Worker = &worker{}

type worker struct {
	WorkerConfig
	Handler
	STS.Client

	create func(int64, TaskRequest) Task
	Tasks  []Task

	countReady int64
	countAll   int64

	serial *sync.Mutex
	*sync.Mutex
}

// NewWorker ...
func NewWorker(
	conf WorkerConfig,
	handler Handler,
	sts STS.Client,
	create func(int64, TaskRequest) Task,
) *worker {
	return &worker{
		WorkerConfig: conf,
		Handler:      handler,
		Client:       sts,
		create:       create,
		Tasks:        make([]Task, 0),
		serial:       new(sync.Mutex),
		Mutex:        new(sync.Mutex),
	}
}

func (w *worker) preDo(ctx context.Context, req TaskRequest) (func() error, error) {
	var (
		ss       = make([]Stream, 0)
		genClean = func(ss []Stream) func() error {
			return func() (err error) {
				for _, s := range ss {
					if err1 := s.Clean(); err1 != nil && err == nil {
						err = err1
					}
				}
				return
			}
		}

		f1 = func(uri interface{}) Stream {
			switch v := uri.(type) {
			case BYTES:
				return NewMem(v)
			case STRING:
				return newstsStream(w.Client, v.String())
			default:
				return nil
			}
		}
		f2 = func(uri interface{}) (interface{}, Stream) {
			switch v := uri.(type) {
			case string:
				return STRING(v), NewFile(v)
			case BYTES:
				return v, NewMem(v)
			default:
				return nil, nil
			}
		}
		f3 = func(uri interface{}) Stream {
			switch v := uri.(type) {
			case BYTES:
				return NewMem(v)
			case STRING:
				return NewFile(v.String())
			default:
				return nil
			}
		}
	)
	switch _req := req.Request().(type) {
	case EvalRequestInner:
		var streams = []Stream{f1(_req.Data.URI)}
		uris, err := w.LoadEval(ctx, streams)
		if err != nil {
			return genClean(ss), err
		}
		var s Stream
		_req.Data.URI, s = f2(uris[0])
		ss = append(ss, s)

		_req, err = w.PreEval(ctx, _req)
		if err != nil {
			return genClean(ss), err
		}
		ss = append(ss, f3(_req.Data.URI))
		req.Reset(newTaskRequest(_req))
	case GroupEvalRequestInner:
		var streams = [][]Stream{make([]Stream, 0)}
		for _, data := range _req.Data {
			streams[0] = append(streams[0], f1(data.URI))
		}
		uris, err := w.LoadGroupEval(ctx, streams)
		if err != nil {
			return genClean(ss), err
		}
		for i := range _req.Data {
			var s Stream
			_req.Data[i].URI, s = f2(uris[0][i])
			ss = append(ss, s)
		}

		_req, err = w.PreGroupEval(ctx, _req)
		if err != nil {
			return genClean(ss), err
		}
		for _, data := range _req.Data {
			ss = append(ss, f3(data.URI))
		}
		req.Reset(newTaskRequest(_req))
	case []EvalRequestInner:
		var (
			reqs    = req.Request().([]EvalRequestInner)
			streams = make([]Stream, 0, len(reqs))
		)
		for _, _req := range reqs {
			streams = append(streams, f1(_req.Data.URI))
		}
		uris, err := w.LoadEval(ctx, streams)
		if err != nil {
			return genClean(ss), err
		}
		for i, name := range uris {
			var s Stream
			reqs[i].Data.URI, s = f2(name)
			ss = append(ss, s)
		}

		for i, _req := range reqs {
			_req, err = w.PreEval(ctx, _req)
			if err != nil {
				return genClean(ss), err
			}
			ss = append(ss, f3(_req.Data.URI))
			reqs[i] = _req
		}
		req.Reset(newTaskRequest(reqs))
	case []GroupEvalRequestInner:
		var (
			reqs    = req.Request().([]GroupEvalRequestInner)
			streams = make([][]Stream, 0, len(reqs))
		)
		for _, _req := range reqs {
			sa := make([]Stream, 0, len(_req.Data))
			for _, data := range _req.Data {
				sa = append(sa, f1(data.URI))
			}
			streams = append(streams, sa)
		}
		uris, err := w.LoadGroupEval(ctx, streams)
		if err != nil {
			return genClean(ss), err
		}
		for i, _req := range reqs {
			for j := range _req.Data {
				var s Stream
				reqs[i].Data[j].URI, s = f2(uris[i][j])
				ss = append(ss, s)
			}
		}

		for i, _req := range reqs {
			_req, err = w.PreGroupEval(ctx, _req)
			if err != nil {
				return genClean(ss), err
			}
			for _, data := range _req.Data {
				ss = append(ss, f3(data.URI))
			}
			reqs[i] = _req
		}
		req.Reset(newTaskRequest(reqs))
	}
	return genClean(ss), nil
}

func (w *worker) Do(ctx context.Context, req TaskRequest, caller ResponseCaller) (bool, error) {

	xl := xlog.FromContextSafe(ctx)
	defer func(b0 time.Time) {
		d0 := time.Since(b0)
		responseTime().
			WithLabelValues("worker.do", "", strconv.Itoa(req.BatchSize())).
			Observe(float64(d0) / 1e9)
		xl.Xprof2("ED", d0, nil)
		xl.Infof("do done. %s", d0)
	}(time.Now())

	ok, _task := func() (bool, Task) { // register
		if atomic.LoadInt64(&w.countAll) >= atomic.LoadInt64(&w.MaxConcurrent) {
			return false, nil
		}
		var (
			id    = time.Now().UnixNano()
			_task = w.create(id, req)
		)
		xl.Infof("do register. %d %#v", id, req)
		atomic.AddInt64(&w.countAll, int64(req.BatchSize()))
		return true, _task
	}()
	if !ok {
		return ok, nil
	}
	defer _task.Close()

	defer atomic.AddInt64(&w.countAll, -1*int64(req.BatchSize()))

	b2 := time.Now()
	var (
		clean func() error
		err   error
	)
	if clean, err = w.preDo(ctx, req); err != nil {
		if err1 := clean(); err1 != nil {
			xl.Warnf("clean local failed. %v", err1)
		}
		xl.Xprof2("EPD", time.Since(b2), err)
		xl.Errorf("pre do failed. %d %s", _task.ID(), err)
		caller.SetHeader(err)
		if err1 := caller.Call(ctx, nil); err1 != nil {
			xl.Warnf("call failed. %v", err1)
		}
		return true, err
	}
	defer func() {
		if err := clean(); err != nil {
			xl.Warnf("clean local failed. %v", err)
		}
	}()

	d2 := time.Since(b2)
	responseTime().
		WithLabelValues("worker.predo", "", strconv.Itoa(req.BatchSize())).
		Observe(float64(d2) / 1e9)
	xl.Xprof2("EPD", d2, nil)
	xl.Infof("predo done. %s", d2)

	atomic.AddInt64(&w.countReady, int64(req.BatchSize()))
	_task.Ready()
	func() {
		w.Lock()
		defer w.Unlock()

		w.Tasks = append(w.Tasks, _task)
	}()

	if atomic.LoadInt64(&w.countReady) >= atomic.LoadInt64(&w.BatchSize) {
		go w.Run(xlog.NewContext(ctx, xl.Spawn()), _task)
	} else {
		go func() {
			_ = <-time.After(time.Duration(atomic.LoadInt64(&w.Wait)))
			w.Run(xlog.NewContext(ctx, xl.Spawn()), _task)
		}()
	}

	b3 := time.Now()
	select {
	case result := <-_task.Watch():
		d3 := time.Since(b3)
		responseTime().
			WithLabelValues("worker.wait4done", formatError(result.err), strconv.Itoa(req.BatchSize())).
			Observe(float64(d3) / 1e9)
		xl.Xprof2("ED", d3, result.err)
		b4 := time.Now()
		if err := w.doCallback(ctx, caller, result); err != nil {
			xl.Warnf("callback failed. %d %s", _task.ID(), err)
		}
		xl.Infof("result. %d %#v %s %s %s", _task.ID(), result, d2, d3, time.Since(b4))
		return true, nil
	case _ = <-ctx.Done():
		err = ctx.Err()
		caller.SetHeader(err)
		caller.Call(context.Background(), nil) // TODO err
		xl.Warnf("do cancel. %s", time.Since(b3))
		return true, err
	}
}

func (w *worker) Run(ctx context.Context, task Task) {

	w.Lock()
	var ok = false
	for _, _task := range w.Tasks {
		if _task.ID() == task.ID() {
			ok = true
			break
		}
	}
	w.Unlock()
	if !ok {
		return
	}
	ok = false

	b1 := time.Now()
	w.serial.Lock()
	defer w.serial.Unlock()

	d1 := time.Since(b1)

	w.Lock()
	for _, _task := range w.Tasks {
		if _task.ID() == task.ID() {
			ok = true
			break
		}
	}
	if !ok {
		w.Unlock()
		return
	}

	var (
		xl    = xlog.FromContextSafe(ctx)
		tasks = []Task{task}
		nReq  = 1
		size  = task.BatchSize()

		indexes = make([]int, 0, len(w.Tasks))
	)
	for j, _task := range w.Tasks {
		if _task.ID() == task.ID() {
			indexes = append(indexes, j)
			continue
		}
		if !_task.IsReady() {
			continue
		}
		if size+_task.BatchSize() <= int(atomic.LoadInt64(&w.BatchSize)) {
			tasks = append(tasks, _task)
			indexes = append(indexes, j)
			size += _task.BatchSize()
			nReq++
		}
	}
	var tasks2 = make([]Task, 0, len(w.Tasks))
	var i = 0
	for j, _task := range w.Tasks {
		if i >= len(indexes) {
			tasks2 = append(tasks2, _task)
			continue
		}
		if indexes[i] == j {
			i++
			continue
		}
		tasks2 = append(tasks2, _task)
	}
	w.Tasks = tasks2
	atomic.AddInt64(&w.countReady, -1*int64(size))
	w.Unlock()

	responseTime().
		WithLabelValues("worker.wait4exec", formatError(nil), strconv.Itoa(size)).
		Observe(float64(d1) / 1e9)
	xl.Infof("wait from run: %s", d1)
	runBatchSize().WithLabelValues().Observe(float64(size))

	xl.Infof("RUN: REQ: %d TASK: %d", nReq, size)

	b2 := time.Now()
	results := w.run(ctx, tasks)
	for i, _task := range tasks {
		_task.Done(results[i])
	}
	d2 := time.Since(b2)
	responseTime().
		WithLabelValues("worker.run", formatError(nil), strconv.Itoa(size)).
		Observe(float64(d2) / 1e9)
	xl.Infof("RUN DONE. REQ: %d TASK: %d %s", nReq, size, d2)
}

func (w *worker) run(ctx context.Context, tasks []Task) []TaskResult {
	var (
		f    func(context.Context, []interface{}) ([]EvalResponseInner, error)
		reqs = make([]interface{}, 0)

		ns = make([][]int, len(tasks))
		as = make([]bool, len(tasks))

		f1 = func(ctx context.Context, reqs []interface{}) ([]EvalResponseInner, error) {
			var reqs2 = make([]EvalRequestInner, 0, len(reqs))
			for _, req := range reqs {
				reqs2 = append(reqs2, req.(EvalRequestInner))
			}
			return w.Eval(ctx, reqs2)
		}
		f2 = func(ctx context.Context, reqs []interface{}) ([]EvalResponseInner, error) {
			var reqs2 = make([]GroupEvalRequestInner, 0, len(reqs))
			for _, req := range reqs {
				reqs2 = append(reqs2, req.(GroupEvalRequestInner))
			}
			return w.GroupEval(ctx, reqs2)
		}
	)
	for i, task := range tasks {
		ns[i] = make([]int, 0)
		switch req := task.Request().(type) {
		case EvalRequestInner:
			if f == nil {
				f = f1
			}
			ns[i] = append(ns[i], len(reqs))
			reqs = append(reqs, req)
		case GroupEvalRequestInner:
			if f == nil {
				f = f2
			}
			ns[i] = append(ns[i], len(reqs))
			reqs = append(reqs, req)
		case []EvalRequestInner:
			if f == nil {
				f = f1
			}
			as[i] = true
			_reqs := task.Request().([]EvalRequestInner)
			for _, req := range _reqs {
				ns[i] = append(ns[i], len(reqs))
				reqs = append(reqs, req)
			}
		case []GroupEvalRequestInner:
			if f == nil {
				f = f2
			}
			as[i] = true
			_reqs := task.Request().([]GroupEvalRequestInner)
			for _, req := range _reqs {
				ns[i] = append(ns[i], len(reqs))
				reqs = append(reqs, req)
			}
		}
	}
	var (
		resps   []EvalResponseInner
		results = make([]TaskResult, len(tasks))
		err     error
	)
	resps, err = f(ctx, reqs)
	for i := range tasks {
		if err == nil {
			if len(resps) < len(reqs) && len(resps) > 0 { // 兼容逻辑，其他Batch错误处理未及时更改完成
				if err1 := detectEvalError(resps[0].Code, resps[0].Message); err1 != nil {
					results[i] = TaskResult{
						IsBatch: as[i],
						err:     err1,
					}
					continue
				}
			}
			results[i] = TaskResult{
				IsBatch: as[i],
			}
			if as[i] {
				results[i].Responses = make([]EvalResponseInner, 0, len(ns[i]))
				for _, j := range ns[i] {
					results[i].Responses = append(results[i].Responses, resps[j])
					if results[i].err == nil {
						if err1 := detectEvalError(resps[j].Code, resps[j].Message); err1 != nil {
							results[i].err = err1
						}
					}
				}
			} else {
				resp := resps[ns[i][0]]
				results[i].Responses = []EvalResponseInner{resp}
				if results[i].err == nil {
					if err1 := detectEvalError(resp.Code, resp.Message); err1 != nil {
						results[i].err = err1
					}
				}
			}
		} else {
			results[i] = TaskResult{
				IsBatch: as[i],
				err:     err,
			}
		}
	}
	return results
}

func (w *worker) doCallback(ctx context.Context, caller ResponseCaller, result TaskResult) error {
	xl := xlog.FromContextSafe(ctx)
	if result.err != nil {
		xl.Errorf("callback: err: %s", result.err)
		caller.SetHeader(result.err)
		return caller.Call(ctx, nil)
	}
	var err error
	if !result.IsBatch {
		if result.Responses[0].Stream != nil {
			var (
				ctx2        = xlog.NewContext(ctx, xl.Spawn())
				rc, size, _ = result.Responses[0].Stream.Open(ctx2)
				clean       = func() { result.Responses[0].Stream.Clean() }
				uri, _      = w.Client.NewURL(ctx, &size)
			)
			wg := new(sync.WaitGroup)
			wg.Add(1)
			done := func(err error) { wg.Done() }
			go func(ctx context.Context) {
				defer rc.Close()
				err := w.SyncPost(ctx, uri, size, rc, done)
				if err != nil {
					xl.Error("doCallback SyncPost", err, uri)
				}
				clean()
			}(util.SpawnContextOnlyReqID(ctx2))
			wg.Wait()
			MergeHeader(caller.Header(), result.Responses[0].Header)
			caller.Header().Set("Content-Type", "application/octet-stream")
			caller.Header().Set("Content-Length", strconv.FormatInt(size, 10))
			err = caller.Call(ctx, uri)
		} else {
			xl.Infof("%#v", result.Responses)
			MergeHeader(caller.Header(), result.Responses[0].Header)
			caller.Header().Set("Content-Type", "application/json")
			if result.Responses == nil || len(result.Responses) == 0 {
				err = caller.Call(ctx, EvalResponse{Code: 599})
			} else {
				err = caller.Call(ctx, EvalResponse{
					Code:    result.Responses[0].Code,
					Message: result.Responses[0].Message,
					Result:  result.Responses[0].Result,
				})
			}
		}
		return err
	}

	// TODO 暂不支持多返回值

	resps := make([]EvalResponse, 0)
	for i := range result.Responses {
		MergeHeader(caller.Header(), result.Responses[i].Header)
		resps = append(resps, EvalResponse{
			Code:    result.Responses[i].Code,
			Message: result.Responses[i].Message,
			Result:  result.Responses[i].Result,
		})
	}
	caller.Header().Set("Content-Type", "application/json")
	return caller.Call(ctx, resps)
}
