package service

import (
	"context"
	"encoding/csv"
	"fmt"
	"io/ioutil"
	"net/http"
	"path/filepath"
	"strings"
	"time"

	"github.com/qiniu/http/httputil.v1"
	"github.com/qiniu/uuid"
	xlog "github.com/qiniu/xlog.v1"
	"qiniu.com/argus/dbstorage/dao"
	"qiniu.com/argus/dbstorage/job"
	outer "qiniu.com/argus/dbstorage/outer_service"
	"qiniu.com/argus/dbstorage/proto"
	"qiniu.com/argus/dbstorage/source"
	authstub "qiniu.com/auth/authstub.v1"
)

const DEFAULT_LOG_LIST_LIMIT = 1000

var _ ITaskService = new(TaskService)

type TaskService struct {
	dao        dao.IDao
	config     *proto.TaskServiceConfig
	dispatcher *job.Dispatcher
}

func NewTaskService(dao dao.IDao, config *proto.TaskServiceConfig) ITaskService {
	return &TaskService{
		dao:        dao,
		config:     config,
		dispatcher: job.NewDispatcher(dao, config),
	}
}

func (s *TaskService) initContext(ctx context.Context, env *authstub.Env) (*xlog.Logger, context.Context) {
	xl, ok := xlog.FromContext(ctx)
	if !ok {
		xl = xlog.New(env.W, env.Req)
		ctx = xlog.NewContext(ctx, xl)
	}
	return xl, ctx
}

func (s *TaskService) getFormConfig(env *authstub.Env) (config proto.TaskConfig, err error) {
	if config.RejectBadFace, err = proto.GetValidBool(env.Req.FormValue("reject_bad_face")); err != nil {
		err = proto.ErrInvalidRejectBadFace
		return
	}

	if config.Mode, err = proto.GetValidMode(env.Req.FormValue("mode")); err != nil {
		err = proto.ErrInvalidMode
		return
	}

	return config, nil
}

func (s *TaskService) PostTaskNew(ctx context.Context, env *authstub.Env) (*TaskNewRespItem, error) {
	xl, ctx := s.initContext(ctx, env)

	groupName := env.Req.FormValue("group_name")
	if groupName == "" {
		return nil, proto.ErrEmptyGroupName
	}

	config, err := s.getFormConfig(env)
	if err != nil {
		xl.Errorf("new task => parse config error: %s", err)
		return nil, err
	}

	//get uploaded file & calculate count
	formFile, info, err := env.Req.FormFile("file")
	if err != nil {
		xl.Error("file is not uploaded : ", err)
		return nil, proto.ErrInvalidFile
	}
	defer formFile.Close()

	content, err := ioutil.ReadAll(formFile)
	if err != nil {
		xl.Error("read uploaded file err : ", err)
		return nil, proto.ErrInvalidFile
	}

	lineNum, ext, err := checkFile(ctx, info.Filename, content)
	if err != nil {
		return nil, err
	}

	//create task
	taskId, err := uuid.Gen(16)
	if err != nil {
		xl.Errorf("fail to generate taskid: %s", err)
		return nil, proto.ErrCreateTaskFail
	}

	err = s.dao.NewTaskFile(ctx, taskId, content)
	if err != nil {
		xl.Errorf("dao.NewTaskFile fail: %s", err)
		return nil, proto.ErrCreateTaskFail
	}

	err = s.dao.NewTaskLog(ctx, proto.TaskId(taskId), env.UserInfo.Uid)
	if err != nil {
		xl.Errorf("dao.NewTaskLog fail: %s", err)
		return nil, proto.ErrCreateTaskFail
	}

	err = s.dao.NewTask(ctx, proto.TaskId(taskId), proto.GroupName(groupName), config, lineNum, ext, env.UserInfo.Uid)
	if err != nil {
		xl.Errorf("dao.NewTask fail: %s", err)
		return nil, proto.ErrCreateTaskFail
	}

	return &TaskNewRespItem{Id: proto.TaskId(taskId)}, nil
}

func (s *TaskService) PostTask_Start(ctx context.Context, args *struct{ CmdArgs []string }, env *authstub.Env) error {
	xl, ctx := s.initContext(ctx, env)

	taskId := args.CmdArgs[0]
	if taskId == "" {
		return proto.ErrInvalidArgument
	}

	//get task
	task, err := s.dao.GetTask(ctx, proto.TaskId(taskId), env.UserInfo.Uid)
	if err != nil {
		xl.Errorf("dao.GetTask fail: %s", err)
		return httputil.NewError(http.StatusInternalServerError, err.Error())
	}

	//validate task status
	if task.Status == proto.RUNNING || task.Status == proto.PENDING || task.Status == proto.STOPPING {
		xl.Error("task is already started")
		return proto.ErrTaskAlreadyStarted
	} else if task.Status == proto.COMPLETED {
		xl.Error("task is already completed")
		return proto.ErrTaskAlreadyCompleted
	}

	//get task file content
	content, err := s.dao.GetTaskFile(ctx, task.FileName)
	if err != nil {
		xl.Errorf("dao.GetTaskFile fail: %s", err)
		return proto.ErrTaskNotExist
	}

	var src source.ISource
	ext := strings.ToLower(task.FileExt)
	switch ext {
	case ".csv":
		src = source.NewCsvSource(content)
	case ".json":
		src = source.NewJsonSource(content)
	default:
		return proto.ErrInvalidFileType
	}

	//init face group service
	outerService := outer.NewFaceGroup(
		s.config.FeatureGroupService.Host,
		s.config.FeatureGroupService.Timeout*time.Second,
		s.config.IsPrivate,
		env.UserInfo.Uid,
		env.UserInfo.Utype,
	)

	//start task
	if err = s.dispatcher.New(ctx, task, src, outerService, true, false); err != nil {
		xl.Errorf("fail to start task, dispatcher.New fail: %s", err)
		return err
	}

	return nil
}
func (s *TaskService) PostTask_Stop(ctx context.Context, args *struct{ CmdArgs []string }, env *authstub.Env) error {
	xl, ctx := s.initContext(ctx, env)

	taskId := args.CmdArgs[0]
	if taskId == "" {
		return proto.ErrInvalidArgument
	}

	//get task
	task, err := s.dao.GetTask(ctx, proto.TaskId(taskId), env.UserInfo.Uid)
	if err != nil {
		xl.Errorf("dao.GetTask fail: %s", err)
		return httputil.NewError(http.StatusInternalServerError, err.Error())
	}

	//validate task status
	if task.Status == proto.STOPPING {
		xl.Error("task is stopping")
		return proto.ErrTaskStopping
	} else if task.Status != proto.RUNNING && task.Status != proto.PENDING {
		xl.Error("task is not running")
		return proto.ErrTaskNotStarted
	}

	//stop task
	if err = s.dispatcher.Stop(ctx, task.TaskId); err != nil {
		xl.Errorf("fail to stop task, dispatcher.Stop fail: %s", err)
		return err
	}
	return nil
}

func (s *TaskService) PostTask_Delete(ctx context.Context, args *struct{ CmdArgs []string }, env *authstub.Env) error {
	xl, ctx := s.initContext(ctx, env)

	taskId := args.CmdArgs[0]
	if taskId == "" {
		return proto.ErrInvalidArgument
	}

	//get task
	task, err := s.dao.GetTask(ctx, proto.TaskId(taskId), env.UserInfo.Uid)
	if err != nil {
		xl.Errorf("dao.GetTask fail: %s", err)
		return httputil.NewError(http.StatusInternalServerError, err.Error())
	}

	//validate task status
	if task.Status == proto.RUNNING || task.Status == proto.PENDING || task.Status == proto.STOPPING {
		xl.Error("task is running")
		return proto.ErrTaskStarted
	}

	//delete task
	if err = s.dao.DeleteTaskLog(ctx, proto.TaskId(taskId)); err != nil {
		xl.Errorf("dao.DeleteTaskLog fail: %s", err)
		return proto.ErrDeleteTaskFail
	}

	if err = s.dao.DeleteTaskFile(ctx, task.FileName); err != nil {
		xl.Errorf("dao.DeleteTaskFile fail: %s", err)
		return proto.ErrDeleteTaskFail
	}

	if err = s.dao.DeleteTask(ctx, proto.TaskId(taskId), env.UserInfo.Uid); err != nil {
		xl.Errorf("dao.DeleteTask fail: %s", err)
		return proto.ErrDeleteTaskFail
	}

	return nil
}

func (s *TaskService) GetTask_Detail(ctx context.Context, args *struct{ CmdArgs []string }, env *authstub.Env) (*TaskDetailRespItem, error) {
	xl, ctx := s.initContext(ctx, env)

	taskId := args.CmdArgs[0]
	if taskId == "" {
		return nil, proto.ErrInvalidArgument
	}

	//get task
	task, err := s.dao.GetTask(ctx, proto.TaskId(taskId), env.UserInfo.Uid)
	if err != nil {
		xl.Errorf("dao.GetTask fail: %s", err)
		return nil, httputil.NewError(http.StatusInternalServerError, err.Error())
	}

	errCount, err := s.dao.GetErrorCount(ctx, proto.TaskId(taskId), env.UserInfo.Uid)
	if err != nil {
		xl.Errorf("dao.GetErrorCount fail: %s", err)
		return nil, httputil.NewError(http.StatusInternalServerError, err.Error())
	}

	return &TaskDetailRespItem{
		TaskId:       task.TaskId,
		GroupName:    task.GroupName,
		Config:       task.Config,
		TotalCount:   task.TotalCount,
		HandledCount: task.HandledCount,
		Status:       task.Status,
		LastError:    task.LastError,
		SuccessCount: task.HandledCount - errCount,
		FailCount:    errCount,
	}, nil
}

func (s *TaskService) GetTask_List(
	ctx context.Context,
	args *struct {
		CmdArgs []string
		Status  proto.TaskStatus `json:"status"`
	},
	env *authstub.Env) (*TaskListRespItem, error) {
	xl, ctx := s.initContext(ctx, env)

	groupName := args.CmdArgs[0]
	if groupName == "" {
		return nil, proto.ErrInvalidArgument
	}

	tasks, err := s.dao.GetTaskList(ctx, proto.GroupName(groupName), proto.TaskStatus(args.Status), env.UserInfo.Uid)
	if err != nil {
		xl.Errorf("dao.GetTaskIds fail: %s", err)
		return nil, httputil.NewError(http.StatusInternalServerError, err.Error())
	}

	res := &TaskListRespItem{Tasks: make([]TaskListRespItemValue, 0)}
	for _, v := range tasks {
		res.Tasks = append(res.Tasks, TaskListRespItemValue{
			Id:     v.TaskId,
			Status: v.Status,
		})
	}
	return res, nil
}

func (s *TaskService) GetTask_Log(
	ctx context.Context,
	args *struct {
		CmdArgs []string
		Skip    int `json:"skip"`
		Limit   int `json:"limit"`
	}, env *authstub.Env) (*TaskLogRespItem, error) {
	xl, ctx := s.initContext(ctx, env)

	taskId := args.CmdArgs[0]
	if taskId == "" {
		return nil, proto.ErrInvalidArgument
	}

	limit := args.Limit
	if limit <= 0 || limit > DEFAULT_LOG_LIST_LIMIT {
		limit = DEFAULT_LOG_LIST_LIMIT
	}

	logs, count, err := s.dao.GetErrorLog(ctx, proto.TaskId(taskId), args.Skip, limit, env.UserInfo.Uid)
	if err != nil {
		xl.Errorf("dao.GetErrorLog fail: %s", err)
		return nil, httputil.NewError(http.StatusInternalServerError, err.Error())
	}

	return &TaskLogRespItem{Logs: logs, TotalCount: count}, nil
}

func (s *TaskService) GetTask_LogDownload(
	ctx context.Context,
	args *struct {
		CmdArgs []string
	},
	env *authstub.Env) {
	xl, ctx := s.initContext(ctx, env)

	taskId := args.CmdArgs[0]
	if taskId == "" {
		httputil.ReplyErr(env.W, proto.ErrInvalidArgument.Code, proto.ErrInvalidArgument.Err)
		return
	}

	logs, _, err := s.dao.GetErrorLog(ctx, proto.TaskId(taskId), 0, 0, env.UserInfo.Uid)
	if err != nil {
		xl.Errorf("dao.GetErrorLog fail: %s", err)
		code, msg := httputil.DetectError(err)
		httputil.ReplyErr(env.W, code, msg)
		return
	}

	env.W.Header().Set("Content-Disposition", "attachment;fileName=error.csv")
	writer := csv.NewWriter(env.W)
	defer writer.Flush()
	for _, v := range logs {
		writer.Write([]string{v.Uri, fmt.Sprintf("%d", v.Code), v.Message})
	}
}

func checkFile(ctx context.Context, name string, content []byte) (int, string, error) {
	xl := xlog.FromContextSafe(ctx)
	//only support csv or json file
	ext := strings.ToLower(filepath.Ext(name))
	if ext != ".csv" && ext != ".json" {
		xl.Error("file type not supported")
		return 0, "", proto.ErrInvalidFileType
	}

	var (
		count int
		err   error
	)
	switch ext {
	case ".csv":
		count, err = source.NewCsvSource(content).GetInfo(ctx)
		if err != nil {
			return 0, "", err
		}
	case ".json":
		count, err = source.NewJsonSource(content).GetInfo(ctx)
		if err != nil {
			return 0, "", err
		}
	}

	if count == 0 {
		xl.Error("file is empty")
		return 0, "", proto.ErrInvalidFile
	}
	return count, ext, nil
}
