package gate

import (
	"context"
	"errors"
	"net/http"
	"sync"
	"time"

	"github.com/qiniu/http/restrpc.v1"
	"github.com/qiniu/xlog.v1"

	"qiniu.com/argus/atserving/model"
)

// Worker ...
type Worker interface {
	Do(
		context.Context,
		time.Duration,
		[]model.TaskReq,
		func(context.Context, string, string, *string, []byte) error,
	) ([]*model.ResponseMessage, error)

	Handle(context.Context, string, *model.ResponseMessage) error
}

var _ Worker = &worker{}

type worker struct {
	m        map[string]chan *model.ResponseMessage
	callback string
	*sync.Mutex
}

// NewWorker ...
func NewWorker(callback string) Worker {
	return &worker{
		m:        make(map[string]chan *model.ResponseMessage),
		callback: callback + "/callback/",
		Mutex:    new(sync.Mutex),
	}
}

func (w *worker) Do(
	ctx context.Context,
	duration time.Duration,
	reqs []model.TaskReq,
	publish func(context.Context, string, string, *string, []byte) error,
) ([]*model.ResponseMessage, error) {

	var (
		xl     = xlog.FromContextSafe(ctx)
		begin  = time.Now()
		header = make(http.Header)
	)

	header.Set(model.KEY_REQID, xl.ReqId())

	var (
		ch        = make(chan *model.ResponseMessage, len(reqs))
		ids       = make([]string, 0, len(reqs))
		m         = make(map[string]int)
		resps     = make([]*model.ResponseMessage, len(reqs))
		xlogMutex = new(sync.Mutex)
	)
	for i, n := 0, len(reqs); i < n; i++ {
		ids = append(ids, xlog.GenReqId())
		m[ids[i]] = i
	}

	w.Lock()
	for _, id := range ids {
		w.m[id] = ch
	}
	w.Unlock()
	defer func() {
		w.Lock()
		for _, id := range ids {
			delete(w.m, id)
		}
		w.Unlock()
	}()
	ctx, cancel := context.WithTimeout(ctx, duration)
	defer cancel()

	for i, req := range reqs {
		body, _ := model.MarshalRequestMessage(req, ids[i], w.callback, header)
		if err := publish(ctx, ids[i], req.GetCmd(), req.GetVersion(), body); err != nil {
			xl.Errorf("publish failed. %s %s", ids[i], err)
			return nil, err
		}
	}

	for i, n := 0, len(reqs); i < n; i++ {
		select {
		case resp := <-ch:
			xlogMutex.Lock()
			xl.Xput(resp.Header[model.KEY_LOG])
			xl.Xprof("EC", begin, nil)
			xlogMutex.Unlock()
			duration2, _ := time.ParseDuration(resp.Header.Get(model.KEY_DURATION))
			xl.Infof("do. %d %v %v", resp.StatusCode, time.Since(begin), duration2)
			resps[m[resp.ID]] = resp
		case <-ctx.Done():
			err := ctx.Err()
			xl.Warnf("cancel. %v", err)
			if err == context.DeadlineExceeded {
				err = ErrTimeout
			}
			return nil, err
		}
	}
	return resps, nil
}

func (w *worker) Handle(
	ctx context.Context,
	id string,
	resp *model.ResponseMessage,
) error {

	var (
		xl = xlog.FromContextSafe(ctx)
		ch chan *model.ResponseMessage
		ok bool
	)

	w.Lock()
	ch, ok = w.m[id]
	w.Unlock()

	if !ok {
		duration, _ := time.ParseDuration(resp.Header.Get(model.KEY_DURATION))
		xl.Warnf("overdue. %s %s", id, duration)
		return errors.New("overdue")
	}

	resp.ID = id
	ch <- resp

	// NOTHING TO RETURN
	return nil
}

// CallbackReq ...
type CallbackReq struct {
	CmdArgs []string
	ReqBody model.ResponseMessage
}

// POST /callback/<id>
func (w *worker) PostCallback_(ctx context.Context, req *CallbackReq, env *restrpc.Env) error {

	xl, ok := xlog.FromContext(ctx)
	if !ok {
		xl = xlog.NewWithReq(env.Req)
		ctx = xlog.NewContext(ctx, xl)
	}
	//req.ReqBody.Header = env.Req.Header
	xl.Infof("callback. %s %#v", req.CmdArgs[0], req)

	return w.Handle(ctx, req.CmdArgs[0], &req.ReqBody)
}
