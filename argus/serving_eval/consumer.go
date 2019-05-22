package eval

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"strconv"
	"time"

	"github.com/qiniu/xlog.v1"

	. "qiniu.com/argus/atserving/model"
	"qiniu.com/argus/atserving/simplequeue/msq"
	"qiniu.com/argus/atserving/simplequeue/sq"
	SQH "qiniu.com/argus/atserving/simplequeue/sq/handler"
)

// Consumer ...
type Consumer interface {
	StopAndWait()
}

var _ Consumer = &consumer{}

type consumer struct {
	Worker
	*msq.MultiConsumer
}

// NewConsumer ...
func NewConsumer(w Worker, ccs []sq.ConsumerConfig) (*consumer, error) {

	var (
		c   = &consumer{Worker: w}
		_h  = SQH.NewSlowHandler(c, time.Second, time.Second*60) // TODO config
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

	{
		if _m, ok := m.AsNSQMessage(); ok {
			taskAttempts().WithLabelValues().Observe(float64(_m.Attempts))
		}
	}

	var (
		xl   = xlog.FromContextSafe(ctx)
		id   string
		body []byte
		msg  RequestMessage
	)
	defer func() {
		if err := recover(); err != nil {
			xl.Error(err)
			err = nil
		}
	}()

	xl.Infof("handle message. %d", len(m.GetBody()))

	id, body, _ = ParseMessageBody(m.GetBody())
	if err = json.Unmarshal(body, &msg); err != nil {
		xl.Warnf("parse message failed. %s", err)
		return nil
	}

	reqID := msg.Header.Get(KEY_REQID)
	if len(reqID) == 0 {
		reqID = xlog.GenReqId()
	}

	xl.Infof("message. %s %s %s", id, reqID, msg.Callback)

	xl = xlog.NewWith(reqID)
	ctx = xlog.NewContext(ctx, xl)
	var (
		audit = &AuditLog{
			Cmd:       msg.Cmd,
			Version:   msg.Version,
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
		caller = newResponseCaller(msg.Callback, id, audit)
		tr     TaskRequest

		badRequest = func() error {
			caller.SetHeader(ErrBadRequest)
			return caller.Call(ctx, nil)
		}
	)

	req, err := msg.ParseRequest(ctx)
	if err != nil {
		xl.Warnf("bad request. %s", err)
		badRequest()
		return nil
	}

	switch _req := req.(type) {
	case EvalRequest:
		tr = newTaskRequest(ToEvalRequestInner(_req))
	case GroupEvalRequest:
		tr = newTaskRequest(ToGroupEvalRequestInner(_req))
	}

	xl.Infof("message do. %#v", tr)
	begin := time.Now()
	ok, err := c.Do(ctx, tr, caller)
	if !ok {
		xl.Info("too much request")
		return sq.ErrRequeue
	}
	xl.Infof("do request. %s", time.Since(begin))

	if err == ErrRetry {
		return err
	}
	return nil
}

////////////////////////////////////////////////////////////////////////////////

// AuditLog ...
type AuditLog struct {
	Cmd       string
	Version   *string
	ReqHeader http.Header
	ReqBody   interface{}

	Err      error
	Duration time.Duration

	CBAddress  string
	CBID       string
	StatusCode int
	RespHeader http.Header
	RespBody   interface{}
}

// REQ
// 20060102150405000
// PATH = cmd/version
// Request Header
// Request Body
// error
// Callback URL
// Status Code
// Response Header
// Response Body
// Duration

// LogInfo ...
type LogInfo interface {
	Info(...interface{})
}

func (audit *AuditLog) Write(log LogInfo) error {
	b := bytes.NewBuffer(nil)
	b.WriteString("REQ\t")
	b.WriteString(time.Now().Format("20060102150405.000000"))
	b.WriteByte('\t')
	if audit.Version == nil {
		b.WriteString(audit.Cmd)
	} else {
		b.WriteString(audit.Cmd)
		b.WriteByte('/')
		b.WriteString(*audit.Version)
	}
	b.WriteByte('\t')
	bs, _ := json.Marshal(audit.ReqHeader)
	b.Write(bs)
	b.WriteByte('\t')
	bs, _ = json.Marshal(audit.ReqBody)
	b.Write(bs)
	b.WriteByte('\t')
	if audit.Err != nil {
		b.WriteString(audit.Err.Error())
	}
	b.WriteByte('\t')
	b.WriteString(audit.CBAddress + audit.CBID)
	b.WriteByte('\t')
	b.WriteString(strconv.Itoa(audit.StatusCode))
	b.WriteByte('\t')
	bs, _ = json.Marshal(audit.RespHeader)
	b.Write(bs)
	b.WriteByte('\t')
	bs, _ = json.Marshal(audit.RespBody)
	if len(bs) > 512 {
		b.Write(bs[:509])
		b.WriteString("...")
	} else {
		b.Write(bs)
	}
	b.WriteByte('\t')
	b.WriteString(strconv.FormatInt(int64(audit.Duration)/100, 10))
	log.Info(b.String())
	return nil
}
