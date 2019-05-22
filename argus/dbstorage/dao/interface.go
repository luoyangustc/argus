package dao

import (
	"context"

	"qiniu.com/argus/dbstorage/proto"
)

type IDao interface {
	NewTask(context.Context, proto.TaskId, proto.GroupName, proto.TaskConfig, int, string, uint32) error
	DeleteTask(context.Context, proto.TaskId, uint32) error
	GetTask(context.Context, proto.TaskId, uint32) (*proto.Task, error)
	GetTaskList(context.Context, proto.GroupName, proto.TaskStatus, uint32) ([]proto.Task, error)
	ResetTask(context.Context, proto.TaskStatus, ...proto.TaskStatus) error

	UpdateTaskStatus(context.Context, proto.TaskId, proto.TaskStatus) error
	UpdateTaskCount(context.Context, proto.TaskId, int) error
	UpdateTaskError(context.Context, proto.TaskId, proto.TaskError) error

	NewTaskFile(context.Context, string, []byte) error
	DeleteTaskFile(context.Context, string) error
	GetTaskFile(context.Context, string) ([]byte, error)

	NewTaskLog(context.Context, proto.TaskId, uint32) error
	GetTaskLog(context.Context, proto.TaskId) (*proto.Log, error)
	DeleteTaskLog(context.Context, proto.TaskId) error

	UpdateProcess(context.Context, proto.TaskId, []int) error
	UpdateHash(context.Context, proto.TaskId, string) error
	UpdateErrorLog(context.Context, proto.TaskId, proto.ErrorLog) error
	GetErrorLog(context.Context, proto.TaskId, int, int, uint32) ([]proto.ErrorLog, int, error)
	GetErrorCount(context.Context, proto.TaskId, uint32) (int, error)
}
