package job

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
	"time"

	"github.com/qiniu/http/httputil.v1"
	"github.com/qiniu/http/restrpc.v1"
	"github.com/qiniu/rpc.v3"
	"github.com/qiniu/xlog.v1"

	"qiniu.com/argus/atserving/model"
	"qiniu.com/argus/atserving/simplequeue/msq"
	"qiniu.com/argus/atserving/simplequeue/sq"
	SQH "qiniu.com/argus/atserving/simplequeue/sq/handler"
)

var (
	ErrAsyncOverdue = httputil.NewError(http.StatusConflict, "overdue")
)

type AsyncC struct {
	ps           []*sq.Producer
	topicDefault string
	topicOptions map[string]string
	callback     string

	m map[string]func(context.Context, []byte, error)
	*sync.Mutex
}

func NewAsyncC(
	ps []*sq.Producer, topicDefault string, topicOptions map[string]string,
	host, patternPrefix string, mux restrpc.Mux,
) *AsyncC {
	c := &AsyncC{
		ps:           ps,
		topicDefault: topicDefault,
		topicOptions: topicOptions,
		callback:     fmt.Sprintf("%s/%s/callback/", host, patternPrefix),
		m:            make(map[string]func(context.Context, []byte, error)),
		Mutex:        new(sync.Mutex),
	}
	router := restrpc.Router{
		PatternPrefix: patternPrefix,
		Mux:           mux,
	}
	router.Register(c)
	return c
}

func (r *AsyncC) Publish(
	ctx context.Context,
	kind string,
	req []byte,
	hook func(context.Context, []byte, error),
) error {

	var (
		xl     = xlog.FromContextSafe(ctx)
		header = make(http.Header)
	)

	header.Set(model.KEY_REQID, xl.ReqId())

	var (
		id      = xlog.GenReqId()
		body, _ = json.Marshal(
			model.RequestMessage{
				ID:       id,
				Header:   header,
				Request:  base64.StdEncoding.EncodeToString(req),
				Callback: r.callback,
			},
		)

		hook2 = func(ctx context.Context, result []byte, err error) {
			r.Lock()
			delete(r.m, id)
			r.Unlock()
			hook(ctx, result, err)
		}
	)
	r.Lock()
	r.m[id] = hook2
	r.Unlock()

	var topic = r.topicDefault
	if r.topicOptions != nil {
		if top, ok := r.topicOptions[kind]; ok {
			topic = top
		}
	}
	xl.Infof("publish. %s %s %s %d", id, kind, topic, len(body))
	_ = r.ps[0].Publish(topic, model.NewMessageBody(id, body))

	return nil
}

// POST /callback/<id>
func (r *AsyncC) PostCallback_(
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
		id   = req.CmdArgs[0]
		hook func(context.Context, []byte, error)
	)
	xl.Infof("callback. %s", id)

	r.Lock()
	hook, ok = r.m[id]
	r.Unlock()
	if !ok {
		duration, _ := time.ParseDuration(req.ReqBody.Header.Get(model.KEY_DURATION))
		xl.Warnf("overdue. %s %s", id, duration)
		return ErrAsyncOverdue
	}

	if req.ReqBody.StatusCode == 0 || req.ReqBody.StatusCode/100 == 2 {
		bs, _ := base64.StdEncoding.DecodeString(req.ReqBody.Response)
		hook(ctx, bs, nil)
	} else {
		hook(ctx, nil, httputil.NewError(req.ReqBody.StatusCode, req.ReqBody.StatusText))
	}
	return nil
}

////////////////////////////////////////////////////////////////////////////////

type AsyncS struct {
	*msq.MultiConsumer

	run func(context.Context, []byte) ([]byte, error)
}

func NewAsyncS(
	ccs []sq.ConsumerConfig,
	run func(context.Context, []byte) ([]byte, error),
) (*AsyncS, error) {

	var (
		c   = &AsyncS{run: run}
		_h  = SQH.NewSlowHandler(c, time.Second, time.Second*3540) // TODO config, 单资源的最长处理时间；小于队列超时
		hs  = make([]sq.Handler, 0, len(ccs))
		err error
	)
	for i, n := 0, len(ccs); i < n; i++ {
		hs = append(hs, _h)
	}

	c.MultiConsumer, err = msq.NewMultiConsumer(ccs, hs, nil)
	if err != nil {
		return nil, err
	}

	return c, nil
}

func (r *AsyncS) StopAndWait() { r.MultiConsumer.StopAndWait() }

func (r *AsyncS) HandleMessage(ctx context.Context, m sq.Message) (err error) {

	var (
		xl   = xlog.FromContextSafe(ctx)
		id   string
		body []byte
		msg  model.RequestMessage
	)
	defer func() {
		if err0 := recover(); err0 != nil {
			xl.Error(err0)
			err = fmt.Errorf("%v", err0)
		}
	}()
	defer func() { err = nil }()

	xl.Infof("handle message. %d", len(m.GetBody()))

	id, body, _ = model.ParseMessageBody(m.GetBody())
	if err = json.Unmarshal(body, &msg); err != nil {
		xl.Warnf("parse message failed. %s", err)
		return nil
	}
	msg.ID = id

	reqID := msg.Header.Get(model.KEY_REQID)
	if len(reqID) == 0 {
		reqID = xlog.GenReqId()
	}

	xl.Infof("message. %s %s %s", msg.ID, reqID, msg.Callback)

	xl = xlog.NewWith(reqID)
	ctx = xlog.NewContext(ctx, xl)

	var (
		ret []byte
	)
	defer func() {
		var (
			resp = model.ResponseMessage{ID: msg.ID, Header: make(http.Header)}
		)
		if err == nil {
			resp.Response = base64.StdEncoding.EncodeToString(ret)
		} else {
			resp.StatusCode, resp.StatusText = httputil.DetectError(err)
		}

		var (
			url = msg.Callback + msg.ID
		)
		resp.Header[model.KEY_LOG] = xl.Xget()
		err = rpc.DefaultClient.CallWithJson(context.Background(), nil, "POST", url, resp) // 保证callback始终传递
		xl.Infof("call: %s %#v %v", url, resp, err)
	}()

	bs, _ := base64.StdEncoding.DecodeString(msg.Request)
	ret, err = r.run(ctx, bs)

	xl.Infof("end. %d %v", len(ret), err)
	return
}
