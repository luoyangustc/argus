package job

import (
	"context"

	. "qiniu.com/argus/bjob/proto"
)

var _ Task = &_Task{}

type _Task struct {
	_Value []byte
}

func NewTask(value []byte) *_Task             { return &_Task{_Value: value} }
func (t *_Task) Value(context.Context) []byte { return t._Value }

var _ TaskResult = &_TaskResult{}

type _TaskResult struct {
	_Task   Task
	_Result []byte
	_Error  error
}

func NewTaskResult(task Task, value []byte, err error) *_TaskResult {
	return &_TaskResult{_Task: task, _Result: value, _Error: err}
}
func (r *_TaskResult) Task() Task                   { return r._Task }
func (r *_TaskResult) Value(context.Context) []byte { return r._Result }
func (r *_TaskResult) Error() error                 { return r._Error }

////////////////////////////////////////////////////////////////////////////////

var _ JobCreator = MockJobCreator{}

type MockJobCreator struct{}

func (c MockJobCreator) NewMaster(ctx context.Context, req []byte, env Env) (JobMaster, error) {
	return &MockJobMaster{}, nil
}

type MockTask struct {
	Value_ []byte
}

func (t MockTask) Value(ctx context.Context) []byte { return t.Value_ }

var _ TaskResult = MockTaskResult{}

type MockTaskResult struct {
	MockTask
	Result []byte
	Err    error
}

func (r MockTaskResult) Task() Task                       { return r.MockTask }
func (r MockTaskResult) Value(ctx context.Context) []byte { return r.Result }
func (r MockTaskResult) Error() error                     { return r.Err }

var _ JobMaster = MockJobMaster{}

type MockJobMaster struct{}

func (m MockJobMaster) NextTask(context.Context) ([]byte, string, bool) { return nil, "", false }
func (m MockJobMaster) Error(context.Context) error                     { return nil }
func (m MockJobMaster) Stop(context.Context)                            {}
func (m MockJobMaster) AppendResult(context.Context, TaskResult) error  { return nil }
func (m MockJobMaster) Result(context.Context) ([]byte, error)          { return nil, nil }

var _ TaskWorker = MockTaskWorker{}

type MockTaskWorker struct{}

func (w MockTaskWorker) Do(context.Context, Task) ([]byte, error) { return nil, nil }
