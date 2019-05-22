package eval

import (
	"context"
	"errors"
	"net/http"
	"testing"

	"github.com/stretchr/testify/assert"

	. "qiniu.com/argus/atserving/model"
)

var _ Worker = &mockWorker{}

type mockWorker struct {
	ok bool
	tr TaskRequest
}

func (mock *mockWorker) Do(ctx context.Context, tr TaskRequest, caller ResponseCaller) (bool, error) {
	mock.tr = tr
	return mock.ok, nil
}
func (mock *mockWorker) Run(ctx context.Context, task Task) {}

////////////////////////////////////////////////////////////////////////////////

func TestResponseCaller(t *testing.T) {
	c := newResponseCaller("http://127.0.0.1:80/callback", "xx", nil)
	{
		code, text := c.parseError(nil)
		assert.Equal(t, http.StatusOK, code)
		assert.Equal(t, "", text)
	}
	{
		code, _ := c.parseError(ErrBadRequest)
		assert.Equal(t, http.StatusBadRequest, code)
	}
	{
		code, text := c.parseError(errors.New("foo"))
		assert.Equal(t, 599, code)
		assert.Equal(t, "foo", text)
	}
	{
		c.SetHeader(errors.New("foo"))
		msg := c.genMessage("foo")
		assert.Equal(t, "xx", msg.ID)
		assert.Equal(t, 599, msg.StatusCode)
		assert.Equal(t, "foo", msg.StatusText)
		assert.Equal(t, "\"foo\"", msg.Response)
	}
}

////////////////////////////////////////////////////////////////////////////////

func TestTaskRequest(t *testing.T) {
	assert.Equal(t, 1, newTaskRequest(EvalRequestInner{}).BatchSize())
	assert.Equal(t, 1, newTaskRequest(GroupEvalRequestInner{}).BatchSize())
	assert.Equal(t, 2,
		newTaskRequest([]EvalRequestInner{EvalRequestInner{}, EvalRequestInner{}}).BatchSize(),
	)
	assert.Equal(t, 2,
		newTaskRequest(
			[]GroupEvalRequestInner{GroupEvalRequestInner{}, GroupEvalRequestInner{}},
		).BatchSize(),
	)

	{
		req1 := EvalRequestInner{Data: ResourceInner{URI: "xxx"}}
		req2 := newTaskRequest(req1)
		req1.Data.URI = STRING("yyy")
		req2.Reset(newTaskRequest(req1))
		req3 := req2.Request().(EvalRequestInner)
		assert.Equal(t, STRING("yyy"), req3.Data.URI)
	}
	{
		req1 := []EvalRequestInner{{Data: ResourceInner{URI: "xxx"}}}
		req2 := newTaskRequest(req1)
		req1[0].Data.URI = STRING("yyy")
		req3 := req2.Request().([]EvalRequestInner)
		assert.Equal(t, STRING("yyy"), req3[0].Data.URI)
	}
}

////////////////////////////////////////////////////////////////////////////////

func TestWorkerPreDo(t *testing.T) {
	var (
		handler = &mockHandler{}
		sts     = newMockSTS()
		w       = NewWorker(
			WorkerConfig{},
			handler,
			sts,
			func(id int64, tr TaskRequest) Task { return NewTask(id, tr) },
		)
	)
	{
		handler.err = nil
		handler.preEvalRet = []interface{}{"foo2"}
		req := newTaskRequest(EvalRequestInner{Data: ResourceInner{URI: "foo1"}})
		_, err := w.preDo(context.Background(), req)
		assert.NoError(t, err)
		assert.Equal(t, STRING("foo2"), req.Request().(EvalRequestInner).Data.URI)
	}
	{
		handler.err = errors.New("foo")
		handler.preEvalRet = nil
		req := newTaskRequest(EvalRequestInner{Data: ResourceInner{URI: "foo1"}})
		_, err := w.preDo(context.Background(), req)
		assert.Equal(t, "foo", err.Error())
	}
	{
		handler.err = nil
		handler.preGroupEvalRet = [][]interface{}{{"foo2"}}
		req := newTaskRequest(GroupEvalRequestInner{Data: []ResourceInner{ResourceInner{URI: "foo1"}}})
		_, err := w.preDo(context.Background(), req)
		assert.NoError(t, err)
		assert.Equal(t, STRING("foo2"), req.Request().(GroupEvalRequestInner).Data[0].URI)
	}
	{
		handler.err = nil
		handler.preEvalRet = []interface{}{"xxx1", "xxx2", "xxx3"}
		req := newTaskRequest(
			[]EvalRequestInner{
				EvalRequestInner{Data: ResourceInner{URI: "foo1"}},
				EvalRequestInner{Data: ResourceInner{URI: "foo2"}},
				EvalRequestInner{Data: ResourceInner{URI: "foo3"}},
			},
		)
		_, err := w.preDo(context.Background(), req)
		assert.NoError(t, err)
		assert.Equal(t, STRING("xxx1"), req.Request().([]EvalRequestInner)[0].Data.URI)
		assert.Equal(t, STRING("xxx2"), req.Request().([]EvalRequestInner)[1].Data.URI)
		assert.Equal(t, STRING("xxx3"), req.Request().([]EvalRequestInner)[2].Data.URI)
	}
	{
		handler.err = nil
		handler.preGroupEvalRet = [][]interface{}{{"xxx1", "xxx2"}, {"xxx3"}}
		req := newTaskRequest(
			[]GroupEvalRequestInner{
				GroupEvalRequestInner{Data: []ResourceInner{ResourceInner{URI: "foo1"}, ResourceInner{URI: "foo2"}}},
				GroupEvalRequestInner{Data: []ResourceInner{ResourceInner{URI: "foo3"}}},
			},
		)
		_, err := w.preDo(context.Background(), req)
		assert.NoError(t, err)
		assert.Equal(t, STRING("xxx1"), req.Request().([]GroupEvalRequestInner)[0].Data[0].URI)
		assert.Equal(t, STRING("xxx2"), req.Request().([]GroupEvalRequestInner)[0].Data[1].URI)
		assert.Equal(t, STRING("xxx3"), req.Request().([]GroupEvalRequestInner)[1].Data[0].URI)
	}
}

////////////////////////////////////////////////////////////////////////////////

var _ TaskRequest = mockTaskRequest{}

type mockTaskRequest struct {
	Size int
	Req  interface{}
}

func (m mockTaskRequest) BatchSize() int        { return m.Size }
func (m mockTaskRequest) Request() interface{}  { return m.Req }
func (m mockTaskRequest) Reset(req interface{}) {}

func TestTask(t *testing.T) {
	_task := NewTask(100, mockTaskRequest{Size: 10, Req: "xx"})
	assert.Equal(t, int64(100), _task.ID())
	assert.Equal(t, 10, _task.BatchSize())
	assert.Equal(t, "xx", _task.Request())
	assert.Equal(t, false, _task.IsReady())
	_task.Ready()
	assert.True(t, _task.IsReady())
	go _task.Done(TaskResult{err: errors.New("yy")})
	resp := <-_task.Watch()
	assert.Equal(t, "yy", resp.err.Error())
}

////////////////////////////////////////////////////////////////////////////////

func TestWorkerRun(t *testing.T) {

	var (
		handler = &mockHandler{}
		sts     = newMockSTS()
		w       = NewWorker(
			WorkerConfig{},
			handler,
			sts,
			func(id int64, tr TaskRequest) Task { return NewTask(id, tr) },
		)
	)

	assert.NotNil(t, w)
	{
		handler.resps, handler.streams, handler.err = nil, nil, nil
		tasks := []Task{NewTask(1, newTaskRequest(EvalRequestInner{}))}
		handler.resps = []EvalResponseInner{EvalResponseInner{Code: 1}}
		results := w.run(context.Background(), tasks)
		assert.Equal(t, 1, len(results))
		assert.Equal(t, false, results[0].IsBatch)
		assert.NoError(t, results[0].err)
		assert.Equal(t, 1, results[0].Responses[0].Code)
	}
	{
		handler.resps, handler.streams, handler.err = nil, nil, nil
		tasks := []Task{NewTask(1, newTaskRequest(EvalRequestInner{}))}
		handler.err = errors.New("foo")
		results := w.run(context.Background(), tasks)
		assert.Equal(t, 1, len(results))
		assert.Equal(t, false, results[0].IsBatch)
		assert.Equal(t, "foo", results[0].err.Error())
	}
	{
		handler.resps, handler.streams, handler.err = nil, nil, nil
		tasks := []Task{NewTask(1, newTaskRequest(EvalRequestInner{}))}
		handler.resps = []EvalResponseInner{EvalResponseInner{Stream: buffer{name: "foo"}}}
		results := w.run(context.Background(), tasks)
		assert.Equal(t, 1, len(results))
		assert.Equal(t, false, results[0].IsBatch)
		assert.NoError(t, results[0].err)
		assert.Equal(t, "foo", results[0].Responses[0].Stream.Name())
	}
	{
		handler.resps, handler.streams, handler.err = nil, nil, nil
		tasks := []Task{
			NewTask(1, newTaskRequest(EvalRequestInner{})),
			NewTask(2, newTaskRequest([]EvalRequestInner{EvalRequestInner{}, EvalRequestInner{}})),
			NewTask(3, newTaskRequest(EvalRequestInner{})),
		}
		handler.resps = []EvalResponseInner{
			EvalResponseInner{Code: 1},
			EvalResponseInner{Code: 2},
			EvalResponseInner{Code: 3},
			EvalResponseInner{Code: 4},
		}
		results := w.run(context.Background(), tasks)

		assert.Equal(t, 3, len(results))

		assert.Equal(t, false, results[0].IsBatch)
		assert.NoError(t, results[0].err)
		assert.Equal(t, 1, results[0].Responses[0].Code)

		assert.Equal(t, true, results[1].IsBatch)
		assert.NoError(t, results[1].err)
		assert.Equal(t, 2, results[1].Responses[0].Code)
		assert.Equal(t, 3, results[1].Responses[1].Code)

		assert.Equal(t, false, results[2].IsBatch)
		assert.NoError(t, results[2].err)
		assert.Equal(t, 4, results[2].Responses[0].Code)
	}
	{
		handler.resps, handler.streams, handler.err = nil, nil, nil
		tasks := []Task{
			NewTask(1, newTaskRequest(EvalRequestInner{})),
			NewTask(2, newTaskRequest([]EvalRequestInner{EvalRequestInner{}, EvalRequestInner{}})),
			NewTask(3, newTaskRequest(EvalRequestInner{})),
		}
		handler.err = errors.New("foo")
		results := w.run(context.Background(), tasks)

		assert.Equal(t, 3, len(results))

		assert.Equal(t, false, results[0].IsBatch)
		assert.Equal(t, "foo", results[0].err.Error())

		assert.Equal(t, true, results[1].IsBatch)
		assert.Equal(t, "foo", results[1].err.Error())

		assert.Equal(t, false, results[2].IsBatch)
		assert.Equal(t, "foo", results[2].err.Error())
	}
	{
		handler.resps, handler.streams, handler.err = nil, nil, nil
		tasks := []Task{
			NewTask(1, newTaskRequest(EvalRequestInner{})),
			NewTask(2, newTaskRequest([]EvalRequestInner{EvalRequestInner{}, EvalRequestInner{}})),
			NewTask(3, newTaskRequest(EvalRequestInner{})),
		}
		handler.resps = []EvalResponseInner{
			EvalResponseInner{Code: 1, Stream: buffer{name: "foo1"}},
			EvalResponseInner{Code: 2},
			EvalResponseInner{Code: 3, Stream: buffer{name: "foo2"}},
			EvalResponseInner{Code: 4},
		}
		results := w.run(context.Background(), tasks)

		assert.Equal(t, 3, len(results))

		assert.Equal(t, false, results[0].IsBatch)
		assert.NoError(t, results[0].err)
		assert.Equal(t, 1, results[0].Responses[0].Code)
		assert.Equal(t, "foo1", results[0].Responses[0].Stream.Name())

		assert.Equal(t, true, results[1].IsBatch)
		assert.NoError(t, results[1].err)
		assert.Equal(t, 2, results[1].Responses[0].Code)
		assert.Equal(t, 3, results[1].Responses[1].Code)
		assert.Equal(t, "foo2", results[1].Responses[1].Stream.Name())

		assert.Equal(t, false, results[2].IsBatch)
		assert.NoError(t, results[2].err)
		assert.Equal(t, 4, results[2].Responses[0].Code)
	}
	{
		handler.resps, handler.streams, handler.err = nil, nil, nil
		tasks := []Task{
			NewTask(1, newTaskRequest(GroupEvalRequestInner{})),
			NewTask(2, newTaskRequest([]GroupEvalRequestInner{GroupEvalRequestInner{}, GroupEvalRequestInner{}})),
			NewTask(3, newTaskRequest(GroupEvalRequestInner{})),
		}
		handler.resps = []EvalResponseInner{
			EvalResponseInner{Code: 1, Stream: buffer{name: "foo1"}},
			EvalResponseInner{Code: 2},
			EvalResponseInner{Code: 3, Stream: buffer{name: "foo2"}},
			EvalResponseInner{Code: 4},
		}
		results := w.run(context.Background(), tasks)

		assert.Equal(t, 3, len(results))

		assert.Equal(t, false, results[0].IsBatch)
		assert.NoError(t, results[0].err)
		assert.Equal(t, 1, results[0].Responses[0].Code)
		assert.Equal(t, "foo1", results[0].Responses[0].Stream.Name())

		assert.Equal(t, true, results[1].IsBatch)
		assert.NoError(t, results[1].err)
		assert.Equal(t, 2, results[1].Responses[0].Code)
		assert.Equal(t, 3, results[1].Responses[1].Code)
		assert.Equal(t, "foo2", results[1].Responses[1].Stream.Name())

		assert.Equal(t, false, results[2].IsBatch)
		assert.NoError(t, results[2].err)
		assert.Equal(t, 4, results[2].Responses[0].Code)
	}
}

////////////////////////////////////////////////////////////////////////////////

var _ ResponseCaller = &mockResponseCaller{}

type mockResponseCaller struct {
	err    error
	header http.Header
	resp   interface{}
}

func newMockResponseCaller() *mockResponseCaller     { return &mockResponseCaller{header: http.Header{}} }
func (mock *mockResponseCaller) cleanup()            { mock.err, mock.header, mock.resp = nil, http.Header{}, nil }
func (mock *mockResponseCaller) SetHeader(err error) { mock.err = err }
func (mock *mockResponseCaller) Header() http.Header { return mock.header }
func (mock *mockResponseCaller) Call(ctx context.Context, resp interface{}) error {
	mock.resp = resp
	return nil
}

func TestWorkerDoCallback(t *testing.T) {

	var (
		handler = &mockHandler{}
		sts     = newMockSTS()
		w       = NewWorker(
			WorkerConfig{},
			handler,
			sts,
			func(id int64, tr TaskRequest) Task { return NewTask(id, tr) },
		)
		caller = newMockResponseCaller()
	)
	{
		caller.cleanup()
		w.doCallback(context.Background(), caller, TaskResult{err: errors.New("foo")})
		assert.NotNil(t, caller.err)
		assert.Equal(t, "foo", caller.err.Error())

	}
	{
		caller.cleanup()
		w.doCallback(
			context.Background(),
			caller,
			TaskResult{Responses: []EvalResponseInner{
				EvalResponseInner{Stream: buffer{name: "foo", bs: []byte("foo")}}}},
		)
		assert.NoError(t, caller.err)
		assert.Equal(t, "3", caller.header.Get("Content-Length"))
	}
	{
		caller.cleanup()
		w.doCallback(
			context.Background(),
			caller,
			TaskResult{Responses: []EvalResponseInner{EvalResponseInner{Code: 1}}},
		)
		assert.NoError(t, caller.err)
		assert.Equal(t, 1, caller.resp.(EvalResponse).Code)
	}
	{
		caller.cleanup()
		w.doCallback(
			context.Background(),
			caller,
			TaskResult{
				IsBatch: true,
				Responses: []EvalResponseInner{
					EvalResponseInner{Code: 1},
					EvalResponseInner{Code: 2},
				},
			},
		)
		assert.NoError(t, caller.err)
		assert.Equal(t, 1, caller.resp.([]EvalResponse)[0].Code)
		assert.Equal(t, 2, caller.resp.([]EvalResponse)[1].Code)
	}

}
