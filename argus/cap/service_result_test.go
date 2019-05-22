package cap

import (
	"context"
	"encoding/json"
	"testing"

	mgoutil "github.com/qiniu/db/mgoutil.v3"
	"github.com/stretchr/testify/assert"
	"qiniu.com/argus/cap/dao"
	"qiniu.com/argus/cap/enums"
	"qiniu.com/argus/cap/model"
	"qiniu.com/argus/cap/task"
)

func TestGetJob_Results(t *testing.T) {
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

	batchResult := task.NewBatchResult(jobDao, taskDao)
	service := NewResultService(batchResult, jobDao, taskDao)

	mode := "mode_pulp"
	jobId := "test_get_job_results"
	jobInMgo := dao.JobInMgo{
		JobID:     jobId,
		JobType:   string(enums.BATCH),
		LabelMode: mode,
		MimeType:  string(enums.MimeTypeImage),
		Status:    string(enums.JobBegin),
	}

	taskId := "test_get_job_results_task"
	taskInMgo := dao.TaskInMgo{
		TaskID: taskId,
		JobID:  jobId,
		URI:    "http://testUrl.com",
		Status: string(enums.TaskTodo),
	}

	err = jobDao.Insert(ctx, jobInMgo)
	assert.NoError(t, err)
	err = taskDao.Insert(ctx, mode, taskInMgo)
	assert.NoError(t, err)

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

	taskInMgo.Result, err = json.Marshal(result)
	assert.NoError(t, err)
	taskInMgo.Status = enums.TaskDone

	err = taskDao.Update(ctx, mode, &taskInMgo)
	assert.NoError(t, err)

	var (
		n      = 0
		writer = func(model.TaskModel) error {
			n++
			return nil
		}
	)

	_, err = service.all(ctx, jobInMgo.LabelMode, jobInMgo.JobID, "", 10, 100, writer)
	assert.NoError(t, err)

	err = taskDao.Remove(ctx, mode, taskId)
	assert.NoError(t, err)

	err = jobDao.Remove(ctx, jobId)
	assert.NoError(t, err)
}
