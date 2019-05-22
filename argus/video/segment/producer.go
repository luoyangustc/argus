package segment

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"sync"
	"time"

	"github.com/qiniu/http/httputil.v1"
	"github.com/qiniu/http/restrpc.v1"
	"github.com/qiniu/xlog.v1"

	"qiniu.com/argus/atserving/model"
	"qiniu.com/argus/atserving/simplequeue/sq"
)

const (
	_TopicFirst = "first_argus_segment"
)

var _ Job = &ProducerJob{}

// ProducerJob ...
type ProducerJob struct {
	ch  chan ClipResponse
	err error

	cleanup func()
	token   string
	*sync.Mutex
}

func newProducerJob(cleanup func()) *ProducerJob {
	return &ProducerJob{
		ch:      make(chan ClipResponse),
		cleanup: cleanup,
		Mutex:   new(sync.Mutex),
	}
}

func (job *ProducerJob) Clips() <-chan ClipResponse {
	return job.ch
}

// Error ...
func (job *ProducerJob) Error() error { return job.err }

// Stop ...
func (job *ProducerJob) Stop() {
	job.stop()
}

func (job *ProducerJob) stop() {
	if job.cleanup != nil {
		job.cleanup()
	}
	if job.ch != nil {
		close(job.ch)
	}
}

func (job *ProducerJob) begin(ctx context.Context, token string) error {
	job.Lock()
	defer job.Unlock()
	if job.token != "" {
		return ErrConflictToken
	}
	job.token = token
	return nil
}

func (job *ProducerJob) post(ctx context.Context, token string, resp ClipResponse) error {
	job.Lock()
	defer job.Unlock()
	if job.token != token {
		return ErrConflictToken
	}
	if job.ch != nil {
		job.ch <- resp
	}
	return nil
}

func (job *ProducerJob) end(ctx context.Context, token string, resp EndResponse) {
	job.Lock()
	defer job.Unlock()
	if job.token != token {
		return
	}
	job.stop()
}

var _ Segment = &Producer{}

// Producer ...
type Producer struct {
	ps       []*sq.Producer
	callback string

	m map[string]*ProducerJob
	*sync.Mutex
}

// NewProducer ...
func NewProducer(ps []*sq.Producer, callback string) *Producer {
	return &Producer{
		ps:       ps,
		callback: callback + "/callback/",
		m:        make(map[string]*ProducerJob),
		Mutex:    new(sync.Mutex),
	}
}

func (p *Producer) Run(ctx context.Context, req SegmentRequest) (Job, error) {

	var (
		xl     = xlog.FromContextSafe(ctx)
		header = make(http.Header)
		id     = xlog.GenReqId()
		job    = newProducerJob(func() {
			p.Lock()
			defer p.Unlock()
			delete(p.m, id)
		})
	)

	header.Set(model.KEY_REQID, xl.ReqId())

	p.Lock()
	p.m[id] = job
	p.Unlock()

	bs, _ := json.Marshal(req)
	bs, _ = json.Marshal(
		model.RequestMessage{
			ID:       id,
			Header:   header,
			Request:  string(bs),
			Callback: p.callback,
		},
	)
	if err := p.publish(ctx, id, bs); err != nil {
		xl.Errorf("publish failed. %s %s", id, err)
	}
	return job, nil
}

func (p *Producer) publish(
	ctx context.Context,
	id string,
	body []byte,
) error {
	var (
		xl    = xlog.FromContextSafe(ctx)
		topic = _TopicFirst
	)
	body = model.NewMessageBody(id, body)

	xl.Infof("publish. %s %s %d", id, topic, len(body))
	// TODO RETRY
	return p.ps[0].Publish(topic, body)
}

// POST /callback/begin/<id>/<token>
func (p *Producer) PostCallbackBegin__(
	ctx context.Context,
	req *struct {
		CmdArgs []string
		ReqBody model.ResponseMessage
	},
	env *restrpc.Env,
) error {

	xl, ok := xlog.FromContext(ctx)
	if !ok {
		xl = xlog.NewWithReq(env.Req)
		ctx = xlog.NewContext(ctx, xl)
	}

	var (
		id    = req.CmdArgs[0]
		token = req.CmdArgs[1]
		job   *ProducerJob
	)
	xl.Infof("callback begin. %s %s %#v", id, token, req.ReqBody)

	p.Lock()
	job, ok = p.m[id]
	p.Unlock()
	if !ok {
		duration, _ := time.ParseDuration(req.ReqBody.Header.Get(model.KEY_DURATION))
		xl.Warnf("overdue. %s %s", id, duration)
		return errors.New("overdue")
	}

	err := job.begin(ctx, token)
	xl.Infof("job begin. %s %s %v", id, token, err)
	return err
}

// POST /callback/end/<id>/<token>
func (p *Producer) PostCallbackEnd__(
	ctx context.Context,
	req *struct {
		CmdArgs []string
		ReqBody model.ResponseMessage
	},
	env *restrpc.Env,
) error {

	xl, ok := xlog.FromContext(ctx)
	if !ok {
		xl = xlog.NewWithReq(env.Req)
		ctx = xlog.NewContext(ctx, xl)
	}

	var (
		id    = req.CmdArgs[0]
		token = req.CmdArgs[1]
		job   *ProducerJob
	)
	xl.Infof("callback end. %s %#v", req.CmdArgs[0], req.ReqBody)

	p.Lock()
	job, ok = p.m[id]
	p.Unlock()
	if !ok {
		duration, _ := time.ParseDuration(req.ReqBody.Header.Get(model.KEY_DURATION))
		xl.Warnf("overdue. %s %s", id, duration)
		return errors.New("overdue")
	}

	var resp EndResponse
	if req.ReqBody.StatusCode != http.StatusOK {
		err := httputil.NewError(req.ReqBody.StatusCode, req.ReqBody.StatusText)
		xl.Infof("producer end error: %#v", err)
		job.err = err
		job.end(ctx, token, resp)
		return err
	}

	if err := json.Unmarshal([]byte(req.ReqBody.Response), &resp); err != nil {
		xl.Warnf("umarshal end response failed. %s %v", req.ReqBody.Response, err)
		// TODO
	}
	job.end(ctx, token, resp)
	xl.Infof("job end. %s %s %v", id, token)

	// NOTHING TO RETURN
	return nil
}

// POST /callback/clips/<id>/<token>
func (p *Producer) PostCallbackClips__(
	ctx context.Context,
	req *struct {
		CmdArgs []string
		ReqBody model.ResponseMessage
	},
	env *restrpc.Env,
) error {

	xl, ok := xlog.FromContext(ctx)
	if !ok {
		xl = xlog.NewWithReq(env.Req)
		ctx = xlog.NewContext(ctx, xl)
	}

	var (
		id    = req.CmdArgs[0]
		token = req.CmdArgs[1]
		job   *ProducerJob
	)
	xl.Infof("callback clips. %s %#v", id, token, req.ReqBody)

	p.Lock()
	job, ok = p.m[id]
	p.Unlock()
	if !ok {
		duration, _ := time.ParseDuration(req.ReqBody.Header.Get(model.KEY_DURATION))
		xl.Warnf("overdue. %s %s", id, duration)
		return errors.New("overdue")
	}

	var resp ClipResponse
	if err := json.Unmarshal([]byte(req.ReqBody.Response), &resp); err != nil {
		xl.Warnf("umarshal clips response failed. %s %v", req.ReqBody.Response, err)
		return httputil.NewError(http.StatusBadRequest, err.Error())
	}
	err := job.post(ctx, token, resp)
	xl.Infof("post clips. %s %s %v", id, token, err)
	return err
}
