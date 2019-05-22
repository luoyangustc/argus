package job

import (
	"context"
	"encoding/json"
	"fmt"
	"testing"

	mgoutil "github.com/qiniu/db/mgoutil.v3"
	"github.com/stretchr/testify/assert"
	"qiniu.com/argus/cap/dao"
	"qiniu.com/argus/cap/enums"
	"qiniu.com/argus/cap/model"
)

func TestNewJobAndQueryJobById(t *testing.T) {
	var (
		ctx     = context.Background()
		mgoConf = dao.CapMgoConfig{
			IdleTimeout:  5000000000,
			MgoPoolLimit: 5,
			Mgo: mgoutil.Config{
				Host:           "127.0.0.1:27017",
				DB:             "argus_new_cap",
				Mode:           "strong",
				SyncTimeoutInS: 5,
			},
		}
	)
	jobDao, err := dao.NewJobDao(&mgoConf)
	assert.NoError(t, err)
	taskDao, err := dao.NewTaskDao(&mgoConf)
	assert.NoError(t, err)

	batchResult := NewJobHandler(jobDao, taskDao)

	mode := "mode_pulp"
	jobId := "job_handler_test_job"

	jobModel := model.JobModel{
		JobID:     jobId,
		JobType:   enums.BATCH,
		LabelMode: mode,
		MimeType:  enums.MimeTypeImage,
		Status:    enums.JobBegin,
		Uid:       12345,
	}
	err = batchResult.NewJob(ctx, &jobModel)
	assert.NoError(t, err)

	_, err = batchResult.QueryJobById(ctx, jobId)
	assert.NoError(t, err)

	err = jobDao.Remove(ctx, jobId)
	assert.NoError(t, err)
}
func TestPushTasks(t *testing.T) {
	var (
		ctx     = context.Background()
		mgoConf = dao.CapMgoConfig{
			IdleTimeout:  5000000000,
			MgoPoolLimit: 5,
			Mgo: mgoutil.Config{
				Host:           "127.0.0.1:27017",
				DB:             "argus_new_cap",
				Mode:           "strong",
				SyncTimeoutInS: 5,
			},
		}
	)
	jobDao, err := dao.NewJobDao(&mgoConf)
	assert.NoError(t, err)
	taskDao, err := dao.NewTaskDao(&mgoConf)
	assert.NoError(t, err)

	batchResult := NewJobHandler(jobDao, taskDao)

	mode := "mode_pulp"
	jobId := "test_push_tasks_jobid"
	jobModel := model.JobModel{
		JobID:     jobId,
		JobType:   enums.BATCH,
		LabelMode: mode,
		MimeType:  enums.MimeTypeImage,
		Status:    enums.JobBegin,
		Uid:       12345,
	}
	err = batchResult.NewJob(ctx, &jobModel)
	assert.NoError(t, err)

	//For task
	taskId := "test_push_tasks_taskid"
	taskModel := model.TaskModel{
		TaskID: taskId,
		JobID:  jobId,
		URI:    "http://test",
		Status: "todo",
	}
	//LabelInfo
	label := model.LabelInfo{
		Name: "pulp",
		Type: enums.LableClassification,
		Data: make([]interface{}, 0),
	}
	lData := model.LabelData{
		Class: "normal",
		Score: 0.89,
	}
	label.Data = append(label.Data, lData)

	labels := []model.LabelInfo{label}
	taskModel.Labels = labels

	err = batchResult.PushTasks(ctx, jobId, []model.TaskModel{taskModel})
	assert.NoError(t, err)

	_, err = batchResult.QueryTaskbById(ctx, jobId, taskId)
	assert.NoError(t, err)

	err = taskDao.Remove(ctx, mode, taskId)
	assert.NoError(t, err)

	err = jobDao.Remove(ctx, jobId)
	assert.NoError(t, err)
}

func TestQueryTaskById(t *testing.T) {
	var (
		ctx     = context.Background()
		mgoConf = dao.CapMgoConfig{
			IdleTimeout:  5000000000,
			MgoPoolLimit: 5,
			Mgo: mgoutil.Config{
				Host:           "127.0.0.1:27017",
				DB:             "argus_new_cap",
				Mode:           "strong",
				SyncTimeoutInS: 5,
			},
		}
	)
	jobDao, err := dao.NewJobDao(&mgoConf)
	assert.NoError(t, err)
	taskDao, err := dao.NewTaskDao(&mgoConf)
	assert.NoError(t, err)

	batchResult := NewJobHandler(jobDao, taskDao)

	mode := "mode_pulp"
	jobId := "test_query_task_job"

	jobModel := model.JobModel{
		JobID:     jobId,
		JobType:   enums.BATCH,
		LabelMode: mode,
		MimeType:  enums.MimeTypeImage,
		Status:    enums.JobBegin,
		Uid:       12345,
	}
	err = batchResult.NewJob(ctx, &jobModel)
	assert.NoError(t, err)

	//For task
	taskId := "test_query_task_job_task"
	taskInMgo := dao.TaskInMgo{
		TaskID: taskId,
		JobID:  jobId,
		URI:    "http://testUrl.com",
		Status: string(enums.TaskTodo),
	}

	//LabelInfo
	label := model.LabelInfo{
		Name: "pulp",
		Type: enums.LableClassification,
		Data: make([]interface{}, 0),
	}
	lData := model.LabelData{
		Class: "normal",
		Score: 0.89,
	}
	label.Data = append(label.Data, lData)

	labels := []model.LabelInfo{label}
	//fmt.Printf("label: %#v", label)
	laJson, err := json.Marshal(labels)

	assert.NoError(t, err)
	taskInMgo.Labels = laJson

	err = taskDao.Insert(ctx, mode, taskInMgo)
	assert.NoError(t, err)

	resp, err := batchResult.QueryTaskbById(ctx, jobId, taskId)
	assert.NoError(t, err)
	fmt.Printf("resp.Labels: %#v", resp.Labels)

	//Remove
	err = taskDao.Remove(ctx, mode, taskId)
	assert.NoError(t, err)

	err = jobDao.Remove(ctx, jobId)
	assert.NoError(t, err)
}
