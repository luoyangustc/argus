package vframe

import (
	"context"
	"encoding/json"
	"time"

	"github.com/qiniu/xlog.v1"

	"qiniu.com/argus/atserving/model"
	"qiniu.com/argus/atserving/simplequeue/msq"
	"qiniu.com/argus/atserving/simplequeue/sq"
	SQH "qiniu.com/argus/atserving/simplequeue/sq/handler"
	"qiniu.com/argus/serving_eval"
	STS "qiniu.com/argus/sts/client"
)

// Consumer ...
type Consumer interface {
	StopAndWait()
}

type consumer struct {
	*msq.MultiConsumer
	Vframe
	STS.Client
	_CutHandler
}

// NewConsumer ...
func NewConsumer(
	vf Vframe,
	sts STS.Client,
	ccs []sq.ConsumerConfig,
) (Consumer, error) {

	var (
		c = &consumer{
			Vframe:      vf,
			Client:      sts,
			_CutHandler: newCutHandler(),
		}
		_h  = SQH.NewSlowHandler(c, time.Second, time.Second*3600) // TODO config
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

func (c *consumer) StopAndWait() { c.MultiConsumer.StopAndWait() }

func (c *consumer) HandleMessage(ctx context.Context, m sq.Message) (err error) {

	var (
		xl   = xlog.FromContextSafe(ctx)
		id   string
		body []byte
		msg  model.RequestMessage
	)
	defer func() {
		if err := recover(); err != nil {
			xl.Error(err)
		}
	}()

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

	requestsCounter("SB", "", "").Inc()
	requestsParallel("S").Inc()
	defer func(begin time.Time) {
		requestsParallel("S").Dec()
		responseTimeLong("S", "", "").
			Observe(durationAsFloat64(time.Since(begin)))
	}(time.Now())
	err = c.run(ctx, msg)
	xl.Info("end job ......")
	if err == ErrRetry {
		return err
	}
	return nil
}

func (c *consumer) run(ctx context.Context, msg model.RequestMessage) (err error) {

	var (
		xl = xlog.FromContextSafe(ctx)
	)
	defer func() {
		if err := recover(); err != nil {
			xl.Error(err)
		}
	}()

	var (
		audit = &eval.AuditLog{
			ReqHeader: msg.Header,
			ReqBody:   msg,
			CBAddress: msg.Callback,
			CBID:      msg.ID,
		}
	)

	defer func(begin time.Time) {
		audit.Err = err
		audit.Duration = time.Since(begin)
		audit.Write(xlog.FromContextSafe(ctx))
	}(time.Now())

	var (
		token = xlog.GenReqId()
		begin = newBeginClient(msg.Callback, msg.ID, token)
		end   = newEndResponse(msg.Callback, msg.ID, token, audit)
		cutC  = newCutClient(
			msg.Callback, msg.ID, token,
			c._CutHandler, c.Client, CutClientConfig{},
		)
		req  VframeRequest
		resp EndResponse
	)
	defer func() {
		if err != nil {
			end.SetHeader(err)
			ctx1 := ctx
			if ctx.Err() != nil {
				ctx1 = context.Background()
			}
			if err1 := end.Call(ctx1, nil); err1 != nil {
				xl.Warnf("end call failed. %s %v", msg.Callback, err1)
			}
		} else {
			if err1 := end.Call(ctx, resp); err1 != nil {
				xl.Warnf("end call failed. %s %v", msg.Callback, err1)
			}
		}
	}()

	if err = json.Unmarshal([]byte(msg.Request), &req); err != nil {
		xl.Warnf("unmarshal request failed. %s %v", msg.Request, err)
		err = ErrBadRequest
		return
	}
	if err = begin.Post(ctx); err != nil {
		return
	}

	job, err := c.Run(ctx, req)
	if err != nil {
		xl.Warnf("new vframe failed. %v", err)
		return
	}
	for cut := range job.Cuts() {
		requestsCounter("PostCut", "", "").Inc()
		if err := cutC.Post(spawnContext(ctx), cut); err != nil { // 不使用同一CTX，以防X-Log过长
			xl.Warnf("post cut failed. %v", err)
			job.Stop() // TODO
		}
	}
	if jc, ok := job.(JobClean); ok {
		jc.Clean()
	}
	if err = job.Error(); err != nil {
		xl.Warnf("run vframe failed. %v", err)
	}
	return
}
