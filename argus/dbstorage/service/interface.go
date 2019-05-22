package service

import (
	"context"

	"qiniu.com/argus/dbstorage/proto"
	authstub "qiniu.com/auth/authstub.v1"
)

// 修改此文件请和 docs/Argus/dbstorage.md 同步

type ITaskService interface {
	PostTaskNew(context.Context, *authstub.Env) (*TaskNewRespItem, error)
	PostTask_Start(context.Context, *struct{ CmdArgs []string }, *authstub.Env) error
	PostTask_Stop(context.Context, *struct{ CmdArgs []string }, *authstub.Env) error
	PostTask_Delete(context.Context, *struct{ CmdArgs []string }, *authstub.Env) error
	GetTask_List(
		context.Context,
		*struct {
			CmdArgs []string
			Status  proto.TaskStatus `json:"status"`
		},
		*authstub.Env) (*TaskListRespItem, error)
	GetTask_Detail(context.Context, *struct{ CmdArgs []string }, *authstub.Env) (*TaskDetailRespItem, error)
	GetTask_Log(
		context.Context,
		*struct {
			CmdArgs []string
			Skip    int `json:"skip"`
			Limit   int `json:"limit"`
		},
		*authstub.Env) (*TaskLogRespItem, error)
	GetTask_LogDownload(context.Context, *struct{ CmdArgs []string }, *authstub.Env)
}

type TaskDetailRespItem struct {
	TaskId       proto.TaskId     `json:"task_id"`
	GroupName    proto.GroupName  `json:"group_name"`
	Config       proto.TaskConfig `json:"config"`
	TotalCount   int              `json:"total_count"`
	HandledCount int              `json:"handled_count"`
	SuccessCount int              `json:"success_count"`
	FailCount    int              `json:"fail_count"`
	Status       proto.TaskStatus `json:"status"`
	LastError    proto.TaskError  `json:"last_error"`
}

type TaskLogRespItem struct {
	Logs       []proto.ErrorLog `json:"logs"`
	TotalCount int              `json:"total_count"`
}

type UploadFaceRespItem struct {
	Id string `json:"id"`
}

type TaskNewRespItem struct {
	Id proto.TaskId `json:"id"`
}

type TaskListRespItem struct {
	Tasks []TaskListRespItemValue `json:"tasks"`
}

type TaskListRespItemValue struct {
	Id     proto.TaskId     `json:"id"`
	Status proto.TaskStatus `json:"status"`
}
