package service

import (
	"bytes"
	"context"
	"net/http"
	"testing"

	"github.com/stretchr/testify/assert"
	"qiniu.com/argus/dbstorage/job"
	"qiniu.com/argus/dbstorage/proto"

	authstub "qiniu.com/auth/authstub.v1"
)

type MockDao struct {
	Status       proto.TaskStatus
	GroupName    proto.GroupName
	TotalCount   int
	HandledCount int
	Tasks        []proto.Task
	ErrorLog     []proto.ErrorLog
}

func (d *MockDao) NewTask(ctx context.Context, id proto.TaskId, group proto.GroupName, config proto.TaskConfig, len int, ext string, uid uint32) (err error) {
	return
}
func (d *MockDao) DeleteTask(ctx context.Context, id proto.TaskId, uid uint32) (err error) { return }
func (d *MockDao) GetTask(ctx context.Context, id proto.TaskId, uid uint32) (task *proto.Task, err error) {
	return &proto.Task{
		Status:       d.Status,
		GroupName:    d.GroupName,
		TotalCount:   d.TotalCount,
		HandledCount: d.HandledCount,
	}, nil
}
func (d *MockDao) GetTaskList(ctx context.Context, group proto.GroupName, status proto.TaskStatus, uid uint32) (ids []proto.Task, err error) {
	return d.Tasks, nil
}
func (d *MockDao) ResetTask(ctx context.Context, stat proto.TaskStatus, status ...proto.TaskStatus) (err error) {
	return
}
func (d *MockDao) UpdateTaskStatus(ctx context.Context, id proto.TaskId, status proto.TaskStatus) (err error) {
	return
}
func (d *MockDao) UpdateTaskCount(ctx context.Context, id proto.TaskId, count int) (err error) {
	return
}
func (d *MockDao) UpdateTaskError(ctx context.Context, id proto.TaskId, taskErr proto.TaskError) (err error) {
	return
}
func (d *MockDao) NewTaskFile(ctx context.Context, name string, content []byte) (err error)    { return }
func (d *MockDao) DeleteTaskFile(ctx context.Context, name string) (err error)                 { return }
func (d *MockDao) GetTaskFile(ctx context.Context, name string) (content []byte, err error)    { return }
func (d *MockDao) NewTaskLog(ctx context.Context, id proto.TaskId, uid uint32) (err error)     { return }
func (d *MockDao) GetTaskLog(ctx context.Context, id proto.TaskId) (log *proto.Log, err error) { return }
func (d *MockDao) DeleteTaskLog(ctx context.Context, id proto.TaskId) (err error)              { return }
func (d *MockDao) UpdateProcess(ctx context.Context, id proto.TaskId, process []int) (err error) {
	return
}
func (d *MockDao) UpdateHash(ctx context.Context, id proto.TaskId, hash string) (err error) { return }
func (d *MockDao) UpdateErrorLog(ctx context.Context, id proto.TaskId, log proto.ErrorLog) (err error) {
	return
}
func (d *MockDao) GetErrorLog(ctx context.Context, id proto.TaskId, skip, limit int, uid uint32) (log []proto.ErrorLog, count int, err error) {
	return d.ErrorLog, 0, nil
}
func (d *MockDao) GetErrorCount(ctx context.Context, taskId proto.TaskId, uid uint32) (count int, err error) {
	return
}

func getMockEnv() *authstub.Env {
	req := &http.Request{Header: http.Header{}}
	resp := &MockResponse{}
	return &authstub.Env{Req: req, W: resp}
}

type MockResponse struct{}

func (r *MockResponse) Header() http.Header         { return http.Header{} }
func (r *MockResponse) Write(b []byte) (int, error) { return 0, nil }
func (r *MockResponse) WriteHeader(statusCode int)  {}

func Test_GetFormConfig(t *testing.T) {
	dao := &MockDao{}
	conf := &proto.TaskServiceConfig{}
	s := &TaskService{
		dao:        dao,
		config:     conf,
		dispatcher: job.NewDispatcher(dao, conf),
	}
	env := getMockEnv()
	env.Req.Form = make(map[string][]string)

	env.Req.Form["reject_bad_face"] = []string{"abc"}
	_, err := s.getFormConfig(env)
	assert.Equal(t, proto.ErrInvalidRejectBadFace, err)

	env.Req.Form["reject_bad_face"] = []string{"1"}
	env.Req.Form["mode"] = []string{"SINGLE"}
	config, err := s.getFormConfig(env)
	assert.Nil(t, err)
	assert.Equal(t, true, config.RejectBadFace)
	assert.Equal(t, "SINGLE", config.Mode)
}

func Test_StartTask(t *testing.T) {
	ctx := context.Background()
	dao := &MockDao{}
	env := getMockEnv()
	s := NewTaskService(dao, &proto.TaskServiceConfig{})

	dao.Status = proto.RUNNING
	err := s.PostTask_Start(ctx, &struct{ CmdArgs []string }{CmdArgs: []string{"test123"}}, env)
	assert.Equal(t, proto.ErrTaskAlreadyStarted, err)

	dao.Status = proto.COMPLETED
	err = s.PostTask_Start(ctx, &struct{ CmdArgs []string }{CmdArgs: []string{"test123"}}, env)
	assert.Equal(t, proto.ErrTaskAlreadyCompleted, err)

}

func Test_StopTask(t *testing.T) {
	ctx := context.Background()
	dao := &MockDao{}
	env := getMockEnv()
	s := NewTaskService(dao, &proto.TaskServiceConfig{})

	dao.Status = proto.STOPPING
	err := s.PostTask_Stop(ctx, &struct{ CmdArgs []string }{CmdArgs: []string{"test123"}}, env)
	assert.Equal(t, proto.ErrTaskStopping, err)

	dao.Status = proto.CREATED
	err = s.PostTask_Stop(ctx, &struct{ CmdArgs []string }{CmdArgs: []string{"test123"}}, env)
	assert.Equal(t, proto.ErrTaskNotStarted, err)
}

func Test_DeleteTask(t *testing.T) {
	ctx := context.Background()
	dao := &MockDao{}
	env := getMockEnv()
	s := NewTaskService(dao, &proto.TaskServiceConfig{})

	dao.Status = proto.RUNNING
	err := s.PostTask_Delete(ctx, &struct{ CmdArgs []string }{CmdArgs: []string{"test123"}}, env)
	assert.Equal(t, proto.ErrTaskStarted, err)

	dao.Status = proto.COMPLETED
	err = s.PostTask_Delete(ctx, &struct{ CmdArgs []string }{CmdArgs: []string{"test123"}}, env)
	assert.Nil(t, err)
}

func Test_GetTask(t *testing.T) {
	ctx := context.Background()
	dao := &MockDao{
		Status:       proto.RUNNING,
		TotalCount:   100,
		HandledCount: 100,
		GroupName:    "group",
	}
	env := getMockEnv()
	s := NewTaskService(dao, &proto.TaskServiceConfig{})

	task, err := s.GetTask_Detail(ctx, &struct{ CmdArgs []string }{CmdArgs: []string{"test123"}}, env)
	assert.Nil(t, err)
	assert.Equal(t, proto.RUNNING, task.Status)
	assert.Equal(t, 100, task.TotalCount)
	assert.Equal(t, 100, task.HandledCount)
	assert.Equal(t, "group", string(task.GroupName))
}

func Test_GetTaskList(t *testing.T) {
	ctx := context.Background()
	dao := &MockDao{
		Tasks: []proto.Task{proto.Task{TaskId: "id1"}, proto.Task{TaskId: "id2"}},
	}
	env := getMockEnv()
	s := NewTaskService(dao, &proto.TaskServiceConfig{})

	tasks, err := s.GetTask_List(ctx, &struct {
		CmdArgs []string
		Status  proto.TaskStatus `json:"status"`
	}{CmdArgs: []string{"test123"}}, env)
	assert.Nil(t, err)
	assert.Equal(t, 2, len(tasks.Tasks))
	assert.Equal(t, "id1", string(tasks.Tasks[0].Id))
	assert.Equal(t, "id2", string(tasks.Tasks[1].Id))
}

func Test_GetTask_Log(t *testing.T) {
	ctx := context.Background()
	dao := &MockDao{
		ErrorLog: []proto.ErrorLog{
			proto.ErrorLog{
				Uri:     "uri1",
				Code:    400,
				Message: "bad request",
			},
			proto.ErrorLog{
				Uri:     "uri2",
				Code:    500,
				Message: "system error",
			},
		},
	}
	env := getMockEnv()
	s := NewTaskService(dao, &proto.TaskServiceConfig{})

	s.GetTask_LogDownload(ctx, &struct{ CmdArgs []string }{CmdArgs: []string{"test123"}}, env)
	ret, err := s.GetTask_Log(ctx, &struct {
		CmdArgs []string
		Skip    int `json:"skip"`
		Limit   int `json:"limit"`
	}{CmdArgs: []string{"test123"}, Limit: 2}, env)
	assert.Nil(t, err)
	assert.Equal(t, 2, len(ret.Logs))
	assert.Equal(t, "uri1", ret.Logs[0].Uri)
	assert.Equal(t, 400, int(ret.Logs[0].Code))
	assert.Equal(t, "bad request", ret.Logs[0].Message)

	assert.Equal(t, "uri2", ret.Logs[1].Uri)
	assert.Equal(t, 500, int(ret.Logs[1].Code))
	assert.Equal(t, "system error", ret.Logs[1].Message)
}

func Test_CheckFile(t *testing.T) {
	ctx := context.Background()
	content := []byte("")
	_, _, err := checkFile(ctx, "test.txt", content)
	assert.Equal(t, proto.ErrInvalidFileType, err)

	_, _, err = checkFile(ctx, "test.csv", content)
	assert.Equal(t, proto.ErrInvalidFile, err)

	content = []byte("id1")
	_, _, err = checkFile(ctx, "test.csv", content)
	assert.Equal(t, "csv file of the task must contains at least two columns for id and uri", err.Error())

	content = []byte("id1,uri1,tag1,desc1\nid2,uri2,tag2,desc2")
	count, ext, err := checkFile(ctx, "test.csv", content)
	assert.Nil(t, err)
	assert.Equal(t, 2, count)
	assert.Equal(t, ".csv", ext)

	var buf bytes.Buffer
	buf.WriteString(`abc{"image":{"id":"id1","uri":"uri1"}}`)
	_, _, err = checkFile(ctx, "test.json", buf.Bytes())
	assert.NotNil(t, err)

	var buf2 bytes.Buffer
	buf2.WriteString(`{"image":{"idabc":"id1","uri":"uri1"}}`)
	_, _, err = checkFile(ctx, "test.json", buf2.Bytes())
	assert.NotNil(t, err)

	var buf3 bytes.Buffer
	buf3.WriteString(`{"image":{"id":"id1","uri":"uri1"}}`)
	buf3.WriteString("\n\n")
	buf3.WriteString(`{"image":{"id":"id2","uri":"uri2"}}`)
	count, ext, err = checkFile(ctx, "test.json", buf3.Bytes())
	assert.Nil(t, err)
	assert.Equal(t, 2, count)
	assert.Equal(t, ".json", ext)

}
