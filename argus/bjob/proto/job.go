package proto

import "context"

type Env struct {
	UID   uint32 `json:"uid" bson:"uid"`
	Utype uint32 `json:"utype" bson:"utype"`
	JID   string `json:"-"`
	ReqID string `json:"reqid" bson:"reqid"`
}

type JobCreator interface {
	NewMaster(context.Context, []byte, Env) (JobMaster, error)
}

type JobMaster interface {
	TaskProducer
	TaskResultGatherer
}

type TaskProducer interface {
	NextTask(context.Context) ([]byte, string, bool)
	Error(context.Context) error
	Stop(context.Context)
}

type Task interface {
	Value(context.Context) []byte
}

type TaskResult interface {
	Task() Task
	Error() error
	Value(context.Context) []byte
}

type TaskWorker interface {
	Do(context.Context, Task) ([]byte, error)
}

type TaskResultGatherer interface {
	AppendResult(context.Context, TaskResult) error
	Result(context.Context) ([]byte, error)
}
