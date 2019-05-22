package broker

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/qiniu/log.v1"

	"qiniu.com/dora/sdk/boots/broker/rpc"
)

const (
	opSend = iota
	opPull
	opFinish

	opCount // I'M THE LAST ONE
)

const (
	topicForBalance = "topic_for_balance"
)

const (
	ctFormURLEncode = "application/x-www-form-urlencoded"
	ctJSON          = "application/json"
)

// Config client configurations
type Config struct {
	Hosts     []string
	Transport http.RoundTripper
	Version   string
	logger    *log.Logger

	// retry intervals for the api of finish
	RetryIntervals []time.Duration

	// retry count when encountering the retry condition
	RetryCount int
}

// Client broker api client
type Client struct {
	Config
	rpc.Client

	lb

	sync.RWMutex
	hostIdxOfMsg map[string]uint32

	retrys           []retry
	runRetryInterval time.Duration

	exitChan chan struct{}
	sync.WaitGroup
}

var (
	errEmptyHosts = errors.New("empty hosts")
)

// New create broker api client
func New(cfg *Config) (*Client, error) {
	if len(cfg.Hosts) == 0 {
		return nil, errEmptyHosts
	}
	for i, s := range cfg.Hosts {
		if strings.HasPrefix(s, "http://") {
			continue
		}
		cfg.Hosts[i] = "http://" + s
	}

	tr := cfg.Transport
	if tr == nil {
		tr = http.DefaultTransport
	}

	c := &Client{
		Config:       *cfg,
		Client:       rpc.NewClientWithTransport(tr),
		lb:           newRoundrobin(cfg.Hosts),
		hostIdxOfMsg: make(map[string]uint32, 1024),
		exitChan:     make(chan struct{}),
	}

	if c.logger == nil {
		c.logger = log.Std
	}

	if c.Version == "" {
		c.Version = "v1"
	}

	if len(c.RetryIntervals) == 0 {
		c.RetryIntervals = []time.Duration{
			time.Millisecond * 200,
			time.Second * 2,
			time.Second * 15,
			time.Second * 30,
			time.Minute * 5,
		}
	}

	c.initRetry()
	c.start()
	return c, nil
}

// CreateQueueRequest create queue request
type CreateQueueRequest struct {
	Name       string    `json:"name"`
	Namespace  string    `json:"namespace"`
	Quota      int       `json:"quota"`
	QuotaRatio float32   `json:"quota_ratio"`
	Tags       []TagInfo `json:"tags"`
}

// TagInfo quotas of tag
type TagInfo struct {
	Name       string  `json:"name"`
	Quota      int     `json:"quota"`
	QuotaRatio float32 `json:"quota_ratio"`
}

// CreateQueue create queue
func (c *Client) CreateQueue(ctx context.Context, params *CreateQueueRequest) error {
	_, err := c.CallWithJSON(ctx, nil, params, http.MethodPost, "/queues")
	return err
}

// ViewQueueResponse queue information
type ViewQueueResponse struct {
	Name       string    `json:"name"`
	Namespace  string    `json:"namespace"`
	Quota      int       `json:"quota"`
	QuotaRatio float32   `json:"quota_ratio"`
	Tags       []TagInfo `json:"tags"`
}

// ViewQueue view queue detail
func (c *Client) ViewQueue(ctx context.Context, name string) (q ViewQueueResponse, err error) {
	_, err = c.Call(ctx, &q, http.MethodGet, "/queues/"+name)
	return
}

// ListQueueRequest list queue params
type ListQueueRequest struct {
	Count    int  `json:"count"`
	Offset   int  `json:"offset"`
	HasQuota bool `json:"has_quota"`
}

// ListQueue query queues
func (c *Client) ListQueue(ctx context.Context, params *ListQueueRequest) (
	resp []ViewQueueResponse, err error,
) {
	_, err = c.Call(ctx, &resp, http.MethodGet, "/queues?"+FormMarshal(params).Encode())
	return
}

// UpdateQueueRequest update queue request
type UpdateQueueRequest struct {
	Name       string    `json:"-"`
	Quota      int       `json:"quota"`
	QuotaRatio float32   `json:"quota_ratio"`
	Tags       []TagInfo `json:"tags"`
}

// UpdateQueue update queue
func (c *Client) UpdateQueue(ctx context.Context, params *UpdateQueueRequest) error {
	_, err := c.CallWithJSON(ctx, nil, params, http.MethodPost, "/queues/"+params.Name)
	return err
}

// DeleteQueue create queue
func (c *Client) DeleteQueue(ctx context.Context, name string) (err error) {
	path := "/queues/" + name
	sucCount, err := c.callAll(ctx, nil, nil, path, nil, http.MethodDelete, "")
	if sucCount > 0 {
		err = nil
	}
	return
}

// DeleteJobsOfQueue delete jobs in the queue
func (c *Client) DeleteJobsOfQueue(ctx context.Context, name, tags string) error {
	path := "/queues/" + name + "/jobs?tag=" + tags
	_, err := c.callAll(ctx, nil, nil, path, nil, http.MethodDelete, "")
	return err
}

// ViewQueueBriefResponse view queue brief
type ViewQueueBriefResponse struct {
	Waiting  int `json:"waiting"`
	Doing    int `json:"doing"`
	Finished int `json:"finished"`
}

// ViewQueueBrief the job statistic in the queue
func (c *Client) ViewQueueBrief(ctx context.Context, name, tags string) (
	b ViewQueueBriefResponse,
	err error,
) {
	_, err = c.Call(ctx, &b, http.MethodGet, "/queues/"+name+"/brief?tag="+tags)
	return
}

// Job detail
type Job struct {
	ID     string   `json:"id"`
	Status string   `json:"status"`
	Queue  string   `json:"queue"`
	Tags   []string `json:"tags"`
	Body   string   `json:"body"`
	Suc    bool     `json:"success"`
}

// ViewQueueDetailResponse detail in the queue
type ViewQueueDetailResponse struct {
	Marker string `json:"marker"`
	Jobs   []Job  `json:"jobs"`
}

// ViewQueueDetail the job detail in the queue
func (c *Client) ViewQueueDetail(ctx context.Context, name, marker string, count int) (
	d ViewQueueDetailResponse,
	err error,
) {
	url := fmt.Sprintf("/queues/"+name+"/detail?count=%d", count)
	if marker != "" {
		url += "&marker=" + marker
	}

	_, err = c.Call(ctx, &d, http.MethodGet, url)
	return
}

// Overview contains the information of queue
type Overview struct {
	Name string   `json:"name"`
	Tags []string `json:"tags"`
}

// OverviewResponse overview response
type OverviewResponse struct {
	Overview []Overview `json:"overview"`
}

// Overview overview of namespace
func (c *Client) Overview(ctx context.Context, namespace string) (o OverviewResponse, err error) {
	p, op := fmt.Sprintf("/overview/%s", namespace), &o
	for _, h := range c.Hosts {
		or := OverviewResponse{}
		err = c.callOne(ctx, &or, nil, c.url(h, p), http.MethodGet, "")
		if err != nil {
			c.logger.Errorf("overview from %s %s error %s", h, p, err)
		}

		op = mergeOverview(op, &or)
	}

	if len(op.Overview) > 0 {
		err = nil
	}
	return
}

func mergeOverview(dst, src *OverviewResponse) *OverviewResponse {
	m := make(map[string]*Overview, len(dst.Overview)+len(src.Overview))
	for i := range dst.Overview {
		o := &dst.Overview[i]
		m[o.Name] = o
	}

	for si := range src.Overview {
		so := &src.Overview[si]
		name := so.Name
		op := m[name]
		if op == nil {
			m[name] = so
			continue
		}

		dtags := op.Tags
	NEXT_TAG:
		for _, t := range so.Tags {
			for _, dt := range dtags {
				if t == dt {
					continue NEXT_TAG
				}
			}
			dtags = append(dtags, t)
		}
		op.Tags = dtags
	}

	os, i := make([]Overview, len(m)), 0
	for _, op := range m {
		os[i] = *op
		i++
	}
	dst.Overview = os
	return dst
}

// CreateJobeRequest create job request
type CreateJobeRequest struct {
	Queue string   `json:"queue"`
	Tags  []string `json:"tags"`
	Body  string   `json:"body"`
}

// CreateJobeResponse create job response
type CreateJobeResponse struct {
	ID string `json:"id"`
}

// CreateJob create job
func (c *Client) CreateJob(ctx context.Context, params *CreateJobeRequest) (
	r CreateJobeResponse,
	err error,
) {
	q := params.Queue
	_, err = c.CallWithFormAndQueue(ctx, q, opSend, &r, params, http.MethodPost, "/jobs")
	return
}

// PullJobRequest pull job request
type PullJobRequest struct {
	Queue string   `json:"queue"`
	Tags  []string `json:"tags"`
	Count int      `json:"count"`
}

// PullJobResponse pull job response
type PullJobResponse struct {
	ID    string   `json:"id"`
	State string   `json:"state"`
	Tags  []string `json:"tags"`
	Body  string   `json:"body"`
}

// PullJob pull one job
//
// endeavor to pull the message from all the broker, if no message in one broker, try next one
func (c *Client) PullJob(ctx context.Context, params *PullJobRequest) (
	r []PullJobResponse, err error,
) {
	hc := uint32(len(c.Hosts))
	i := c.selectHostOfQueueAndOp(params.Queue, opPull) % hc
	for e := i + hc; i < e; i++ {
		i, err = c.call(ctx, &r, params, "/jobs", nil, http.MethodPut, ctFormURLEncode, i)
		if err != nil { // tried all host but still error
			return
		}

		if len(r) > 0 {
			for j := range r {
				c.bindHostIdx(i, r[j].ID)
			}
			return
		}
	}
	return
}

// FinishJobRequest finish job request
type FinishJobRequest struct {
	ID         string     `json:"-"`
	Status     string     `json:"status"`
	Result     string     `json:"result"`
	Suc        bool       `json:"success"`
	ReSchedule ReSchedule `json:"reschedule"`
}

// ReSchedule defines reschedule action
type ReSchedule struct {
	Need       bool `json:"need"`
	Deadletter bool `json:"deadletter"`
}

// FinishJob finish job
func (c *Client) FinishJob(ctx context.Context, params *FinishJobRequest) error {
	id := params.ID
	i, exist := c.getHostIdxOf(id)
	if !exist {
		err := fmt.Errorf("finish job but the broker address is not buffered:%s", id)
		c.logger.Error(err)
		return err
	}
	p := "/jobs/" + id
	_, err := c.call(ctx, nil, params, p, nil, http.MethodPost, ctJSON, i)
	c.unbindHostIdx(id)

	if err != nil && needRetry(err) {
		c.retryBackground(c.url(c.hostAt(i), p), params, nil, http.MethodPost, ctJSON)
	}
	return err
}

// DelayJobRequest finish job request
type DelayJobRequest struct {
	ID    string `json:"-"`
	Delay int    `json:"delay"`
}

// DelayJob finish job
func (c *Client) DelayJob(ctx context.Context, params *DelayJobRequest) error {
	id := params.ID
	i, exist := c.getHostIdxOf(id)
	if !exist {
		err := fmt.Errorf("delay job but the broker address is not buffered:%s", id)
		c.logger.Error(err)
		return err
	}
	_, err := c.call(ctx, nil, params, "/jobs/"+id+"/delay", nil, http.MethodPost, ctFormURLEncode, i)
	return err
}

// ViewJobResponse view job response
type ViewJobResponse struct {
	Queue  string   `json:"queue"`
	Tags   []string `json:"tags"`
	Body   string   `json:"body"`
	Status string   `json:"status"`
	ID     string   `json:"id"`
	Result string   `json:"result"`
	Suc    bool     `json:"success"`
}

// ViewJob view job
func (c *Client) ViewJob(ctx context.Context, jobID string) (r ViewJobResponse, err error) {
	_, err = c.Call(ctx, &r, http.MethodGet, "/jobs/%s", jobID)
	return
}

// Change queue changed log
type Change struct {
	Action string `json:"action"`
	Params string `json:"parameters"`
}

// ChangesOfQueueResponse changeing of the queues response
type ChangesOfQueueResponse struct {
	Total   int      `json:"total"`
	Changes []Change `json:"changes"`
}

// ChangesOfQueue changing history of the queues
func (c *Client) ChangesOfQueue(ctx context.Context, offset, count int) (
	r ChangesOfQueueResponse,
	err error,
) {
	_, err = c.Call(ctx, &r, http.MethodGet, "/changes/queue?offset=%d&count=%d", offset, count)
	return
}

// PullJobsFromDeadLetterRequest pull jobs from dead-letter queue
type PullJobsFromDeadLetterRequest struct {
	Queue string   `json:"queue"`
	Tags  []string `json:"tags"`
	From  int      `json:"from"`
	To    int      `json:"to"`
}

// PullJobsFromDeadLetterResponse pull jobs response
type PullJobsFromDeadLetterResponse struct {
	Count int `json:"count"`
}

// PullJobsFromDeadLetter pull jobs from the dead-letter queue
// and can be consumed
func (c *Client) PullJobsFromDeadLetter(ctx context.Context, params *PullJobsFromDeadLetterRequest) (
	r PullJobsFromDeadLetterResponse, err error,
) {
	_, err = c.CallWithForm(ctx, &r, params, http.MethodPost, "/deadletter")
	return
}

// Call detects the graceful down status
func (c *Client) Call(
	ctx context.Context,
	r interface{},
	method, path string,
	urlp ...interface{},
) (uint32, error) {
	return c.call(ctx, r, nil, path, urlp, method, "", c.selectNextHostIndex())
}

// CallWithForm detects the graceful down status
func (c *Client) CallWithForm(
	ctx context.Context,
	r, param interface{},
	method, path string,
	urlp ...interface{},
) (uint32, error) {
	return c.call(ctx, r, param, path, urlp, method, ctFormURLEncode, c.selectNextHostIndex())
}

// CallWithJSON detects the graceful down status
func (c *Client) CallWithJSON(
	ctx context.Context,
	r, param interface{},
	method, path string,
	urlp ...interface{},
) (uint32, error) {
	return c.call(ctx, r, param, path, urlp, method, ctJSON, c.selectNextHostIndex())
}

// CallWithFormAndQueue detects the graceful down status
func (c *Client) CallWithFormAndQueue(
	ctx context.Context,
	queue string,
	op int,
	r, param interface{},
	method, path string,
	urlp ...interface{},
) (uint32, error) {
	return c.call(
		ctx, r, param, path, urlp, method, ctFormURLEncode, c.selectHostOfQueueAndOp(queue, op),
	)
}

// CallWithJSONAndQueue detects the graceful down status
func (c *Client) CallWithJSONAndQueue(
	ctx context.Context,
	queue string,
	op int,
	r, param interface{},
	method, path string,
	urlp ...interface{},
) (uint32, error) {
	return c.call(ctx, r, param, path, urlp, method, ctJSON, c.selectHostOfQueueAndOp(queue, op))
}

// send requests to the brokers
// if broker returns the error which needs to retry, sends the requests to next broker
func (c *Client) call(ctx context.Context,
	ret, param interface{},
	path string, urlp []interface{},
	method, ct string,
	fromHostIdx uint32,
) (hostIdx uint32, err error) {
	hostIdx = fromHostIdx - 1
	for range c.Hosts {
		hostIdx++
		url := c.url(c.hostAt(hostIdx), path, urlp...)
		err = c.callOne(ctx, ret, param, url, method, ct)
		if err == nil {
			return
		}

		if needRetry(err) {
			c.logger.Errorf("call %s to retry since %v \n", url, err)
			continue // TRY NEXT BROKER
		}

		return
	}
	return
}

func (c *Client) url(host, path string, urlp ...interface{}) string {
	return fmt.Sprintf(host+"/"+c.Version+path, urlp...)
}

func (c *Client) callOne(
	ctx context.Context, ret, param interface{}, url, method, ct string,
) error {
	switch ct {
	case ctFormURLEncode:
		body := ""
		if param != nil {
			body = FormMarshal(param).Encode()
		}
		return c.Client.CallWith(ctx, ret, method, url, ct, strings.NewReader(body), len(body))
	case ctJSON:
		return c.Client.CallWithJson(ctx, ret, method, url, param)
	default:
		return c.Client.Call(ctx, ret, method, url)
	}
}

func (c *Client) callAll(ctx context.Context,
	ret, param interface{},
	path string, urlp []interface{},
	method, ct string,
) (sucCount int, err error) {
	tryHosts, retryHosts, retryCount := c.Hosts, make([]string, 0, len(c.Hosts)), c.RetryCount
RETRY:
	for _, host := range tryHosts {
		url := c.url(host, path, urlp...)
		e := c.callOne(ctx, ret, param, url, method, ct)
		if e == nil {
			sucCount++
			continue
		}

		c.logger.Errorf("call %s error %s", url, e)
		err = e
		if needRetry(err) {
			retryHosts = append(retryHosts, host)
		}
	}

	if retryCount == 0 || len(tryHosts) == 0 {
		return
	}
	retryCount--

	tryHosts = retryHosts
	retryHosts = make([]string, 0, len(tryHosts))
	goto RETRY
}

func (c *Client) getHostIdxOf(msgID string) (uint32, bool) {
	c.RLock()
	i, exist := c.hostIdxOfMsg[msgID]
	c.RUnlock()
	return i, exist
}

func (c *Client) bindHostIdx(i uint32, msgID string) {
	c.Lock()
	c.hostIdxOfMsg[msgID] = i
	c.Unlock()
}

func (c *Client) unbindHostIdx(msgID string) {
	c.Lock()
	delete(c.hostIdxOfMsg, msgID)
	c.Unlock()
}

func needRetry(err error) bool {
	e, ok := err.(*rpc.ErrorInfo)
	if !ok {
		return true
	}
	return e.HttpCode() >= 500 && e.HttpCode() < 600
}

type request struct {
	url    string
	param  interface{}
	ret    interface{}
	method string
	ct     string
}

func (r *request) String() string {
	return fmt.Sprintf("url=%s,param=%v,method=%s,ct=%s", r.url, r.param, r.method, r.ct)
}

type retry struct {
	lastTryTime time.Time
	interval    time.Duration
	requests    []request
	c           *Client

	sync.RWMutex
}

func (r *retry) needRun(t time.Time) bool {
	return t.Sub(r.lastTryTime) > r.interval
}

func (r *retry) add(req *request) {
	r.Lock()
	r.requests = append(r.requests, *req)
	r.Unlock()
}

func (r *retry) run() (failed []request) {
	r.Lock()
	hasRequest := len(r.requests) > 0
	var requests []request
	if hasRequest {
		requests = r.requests
		r.requests = make([]request, 0, len(requests))
	}
	r.Unlock()

	if !hasRequest {
		return
	}

	ctx := context.TODO()
	for i := range requests {
		req := &requests[i]
		err := r.c.callOne(ctx, req.ret, req.param, req.url, req.method, req.ct)
		r.c.logger.Infof("retry [%s] failed %v", req, err)

		if err != nil && needRetry(err) {
			failed = append(failed, *req)
		}
	}

	r.lastTryTime = time.Now()
	return
}

func (c *Client) initRetry() {
	intervals := c.RetryIntervals
	c.runRetryInterval = time.Millisecond * 200
	c.retrys = make([]retry, len(intervals))
	for i := range c.retrys {
		r := &c.retrys[i]
		r.interval = intervals[i]
		r.c = c
	}
}

func (c *Client) retryBackground(url string, param, ret interface{}, method, ct string) {
	c.retrys[0].add(&request{url: url, param: param, ret: ret, method: method, ct: ct})
}

func (c *Client) retry() {
	now, l := time.Now(), len(c.retrys)
	for i := l - 1; i != -1; i-- {
		retry := &c.retrys[i]
		if !retry.needRun(now) {
			continue
		}
		failed := retry.run()

		if len(failed) == 0 || i == l-1 {
			continue
		}

		next := &c.retrys[i+1]
		for j := range failed {
			next.add(&failed[j])
		}
	}
}

func (c *Client) startRetryBackground() {
	c.Add(1)
	go func() {
		ticker := time.NewTicker(c.runRetryInterval)
		for {
			select {
			case <-ticker.C:
				c.retry()
			case <-c.exitChan:
				c.retry()
				c.Done()
				ticker.Stop()
				return
			}
		}
	}()
}

func (c *Client) hostAt(i uint32) string {
	return c.Hosts[int(i)%len(c.Hosts)]
}

// start shutdowns the client worker
func (c *Client) start() {
	c.startRetryBackground()
}

// Shutdown shutdowns the client worker
func (c *Client) Shutdown() {
	close(c.exitChan)
	c.Wait()
}
