package eval

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/nsqio/go-nsq"
	"github.com/stretchr/testify/assert"

	. "qiniu.com/argus/atserving/model"
	"qiniu.com/argus/atserving/simplequeue/sq"
)

var _ sq.Message = &mockMessage{}

type mockMessage struct {
	body []byte
}

func (mock *mockMessage) GetBody() []byte              { return mock.body }
func (mock *mockMessage) Touch()                       {}
func (mock *mockMessage) Tag(tag sq.MessageTag) string { return "" }

func (mock *mockMessage) AsNSQMessage() (nsq.Message, bool) { return nsq.Message{}, false }

func TestConsumerHandleMessage(t *testing.T) {

	{
		var (
			w    = &mockWorker{}
			c, _ = NewConsumer(w, []sq.ConsumerConfig{})

			b10   = byte(10)
			id    = "0123456789"
			req   = GroupEvalRequest{Data: []Resource{{URI: "foo"}}}
			bs, _ = json.Marshal(req)
			msg   = RequestMessage{
				Request: string(bs),
			}
		)
		bs, _ = json.Marshal(msg)

		{
			w.ok = false
			err := c.HandleMessage(
				context.Background(),
				&mockMessage{append(append([]byte{b10}, []byte(id)...), bs...)},
			)
			assert.Error(t, sq.ErrRequeue, err)
		}
		{
			w.ok = true
			err := c.HandleMessage(
				context.Background(),
				&mockMessage{append(append([]byte{b10}, []byte(id)...), bs...)},
			)
			assert.NoError(t, err)
			assert.Equal(t, 1, w.tr.BatchSize())
			assert.Equal(t, STRING("foo"), w.tr.Request().(GroupEvalRequestInner).Data[0].URI)
		}
	}
	// 暂不支持
	//{
	//	var (
	//		w    = &mockWorker{}
	//		c, _ = NewConsumer(w, []sq.ConsumerConfig{}, true)
	//
	//		b10 = byte(10)
	//		id  = "0123456789"
	//		req = []GroupEvalRequest{
	//			{Data: []Resource{{URI: "foo1"}, {URI: "foo2"}}},
	//			{Data: []Resource{{URI: "foo3"}}},
	//		}
	//		bs, _ = json.Marshal(req)
	//		msg   = RequestMessage{
	//			Request: string(bs),
	//			Header:  http.Header{KEY_BATCH: []string{}},
	//		}
	//	)
	//	bs, _ = json.Marshal(msg)
	//
	//	w.ok = true
	//	err := c.HandleMessage(
	//		context.Background(),
	//		&mockMessage{append(append([]byte{b10}, []byte(id)...), bs...)},
	//	)
	//	assert.NoError(t, err)
	//	assert.Equal(t, 2, w.tr.BatchSize())
	//	_req := w.tr.Request().([]GroupEvalRequest)
	//	assert.Equal(t, "foo1", _req[0].Data[0].URI)
	//	assert.Equal(t, "foo2", _req[0].Data[1].URI)
	//	assert.Equal(t, "foo3", _req[1].Data[0].URI)
	//}
	{
		var (
			w    = &mockWorker{}
			c, _ = NewConsumer(w, []sq.ConsumerConfig{})

			b10   = byte(10)
			id    = "0123456789"
			req   = EvalRequest{Data: Resource{URI: "foo"}}
			bs, _ = json.Marshal(req)
			msg   = RequestMessage{
				Request: string(bs),
			}
		)
		bs, _ = json.Marshal(msg)

		{
			w.ok = false
			err := c.HandleMessage(
				context.Background(),
				&mockMessage{append(append([]byte{b10}, []byte(id)...), bs...)},
			)
			assert.Error(t, sq.ErrRequeue, err)
		}
		{
			w.ok = true
			err := c.HandleMessage(
				context.Background(),
				&mockMessage{append(append([]byte{b10}, []byte(id)...), bs...)},
			)
			assert.NoError(t, err)
			assert.Equal(t, 1, w.tr.BatchSize())
			assert.Equal(t, STRING("foo"), w.tr.Request().(EvalRequestInner).Data.URI)
		}
	}
	// 暂不支持
	//{
	//	var (
	//		w    = &mockWorker{}
	//		c, _ = NewConsumer(w, []sq.ConsumerConfig{}, false)
	//
	//		b10 = byte(10)
	//		id  = "0123456789"
	//		req = []EvalRequest{
	//			{Data: Resource{URI: "foo1"}},
	//			{Data: Resource{URI: "foo2"}},
	//		}
	//		bs, _ = json.Marshal(req)
	//		msg   = RequestMessage{
	//			Request: string(bs),
	//			Header:  http.Header{KEY_BATCH: []string{}},
	//		}
	//	)
	//	bs, _ = json.Marshal(msg)
	//
	//	w.ok = true
	//	err := c.HandleMessage(
	//		context.Background(),
	//		&mockMessage{append(append([]byte{b10}, []byte(id)...), bs...)},
	//	)
	//	assert.NoError(t, err)
	//	assert.Equal(t, 2, w.tr.BatchSize())
	//	_req := w.tr.Request().([]EvalRequest)
	//	assert.Equal(t, "foo1", _req[0].Data.URI)
	//	assert.Equal(t, "foo2", _req[1].Data.URI)
	//}
}

type mockLogInfo struct {
	output string
}

func (info *mockLogInfo) Info(v ...interface{}) {
	info.output = fmt.Sprint(v...)
}

func TestAuditLog(t *testing.T) {
	var audit = &AuditLog{
		Cmd: "foo",
		ReqHeader: http.Header{
			"X-ReqID": []string{"xxx"},
		},
		ReqBody: struct {
			URI string `json:"uri"`
		}{URI: "http://www.qiniu.com"},

		Duration: time.Second,

		CBAddress:  "http://127.0.0.1/",
		CBID:       "1",
		StatusCode: 200,
		RespHeader: http.Header{
			"X-Log": []string{"yyy"},
		},
		RespBody: struct {
			Result int `json:"result"`
		}{Result: 100},
	}
	var info = &mockLogInfo{}
	audit.Write(info)
	strs := strings.Split(info.output, "\t")
	assert.Equal(t, 11, len(strs))
	assert.Equal(t, "REQ", strs[0])
	assert.Equal(t, "foo", strs[2])
	assert.Equal(t, "{\"X-ReqID\":[\"xxx\"]}", strs[3])
	assert.Equal(t, "{\"uri\":\"http://www.qiniu.com\"}", strs[4])
	assert.Equal(t, "http://127.0.0.1/1", strs[6])
	assert.Equal(t, "200", strs[7])
	assert.Equal(t, "{\"X-Log\":[\"yyy\"]}", strs[8])
	assert.Equal(t, "{\"result\":100}", strs[9])
	assert.Equal(t, "10000000", strs[10])
}
