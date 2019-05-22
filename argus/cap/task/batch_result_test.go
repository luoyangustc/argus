package task

import (
	"context"
	"testing"

	mgoutil "github.com/qiniu/db/mgoutil.v3"
	"github.com/stretchr/testify/assert"
	"qiniu.com/argus/cap/dao"
	"qiniu.com/argus/cap/enums"
	"qiniu.com/argus/cap/model"
)

func TestCheckTasksDone(t *testing.T) {
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

	batchResult := NewBatchResult(jobDao, taskDao)

	mode := "mode_pulp"
	jobId := "test_check_tasks_done_jobid"
	jobInMgo := dao.JobInMgo{
		JobID:     jobId,
		JobType:   string(enums.BATCH),
		LabelMode: mode,
		MimeType:  string(enums.MimeTypeImage),
		Status:    string(enums.JobBegin),
	}

	taskId := "test_check_tasks_done_taskid"
	taskInMgo := dao.TaskInMgo{
		TaskID: taskId,
		JobID:  jobId,
		URI:    "http://testUrl.com",
		Status: string(enums.TaskTodo),
	}

	result := model.TaskModel{
		TaskID: taskId,
		Labels: make([]model.LabelInfo, 0),
	}

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

	result.Labels = append(result.Labels, label)
	label.Name = "terror"
	result.Labels = append(result.Labels, label)

	err = jobDao.Insert(ctx, jobInMgo)
	assert.NoError(t, err)

	err = taskDao.Insert(ctx, mode, taskInMgo)
	assert.NoError(t, err)

	res, err := batchResult.CheckTasksDone(ctx, jobId)
	assert.NoError(t, err)
	assert.Equal(t, false, res)

	taskInMgo.Status = enums.TaskDone
	err = taskDao.Update(ctx, mode, &taskInMgo)
	assert.NoError(t, err)

	res, err = batchResult.CheckTasksDone(ctx, jobId)
	assert.NoError(t, err)
	assert.Equal(t, true, res)

	err = taskDao.Remove(ctx, mode, taskId)
	assert.NoError(t, err)

	err = jobDao.Remove(ctx, jobId)
	assert.NoError(t, err)
}
