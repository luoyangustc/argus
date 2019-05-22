package dao

import (
	"context"
	"testing"

	"qiniu.com/argus/cap/enums"

	mgoutil "github.com/qiniu/db/mgoutil.v3"
	"github.com/stretchr/testify/assert"
)

func TestQueryTasksForAuditor(t *testing.T) {
	var (
		ctx     = context.Background()
		mgoConf = CapMgoConfig{
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

	taskDao, err := NewTaskDao(&mgoConf)
	assert.NoError(t, err)
	_, err = taskDao.FetchTasksForAuditor(ctx, "mode_politician_pulp_terror", 2)

	assert.NoError(t, err)
}

func TestCancelTasks(t *testing.T) {
	var (
		ctx     = context.Background()
		mgoConf = CapMgoConfig{
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

	jobDao, err := NewJobDao(&mgoConf)
	assert.NoError(t, err)

	taskDao, err := NewTaskDao(&mgoConf)
	assert.NoError(t, err)
	mode := "mode_pulp"
	jobId := "test_check_tasks_cancel"
	jobInMgo := JobInMgo{
		JobID:     jobId,
		JobType:   string(enums.BATCH),
		LabelMode: mode,
		MimeType:  string(enums.MimeTypeImage),
		Status:    string(enums.JobBegin),
	}

	taskId := "test_check_tasks_cancel_task"
	taskInMgo := TaskInMgo{
		TaskID: taskId,
		JobID:  jobId,
		URI:    "http://testUrl.com",
		Status: string(enums.TaskDoing),
	}
	err = jobDao.Insert(ctx, jobInMgo)
	assert.NoError(t, err)

	err = taskDao.Insert(ctx, mode, taskInMgo)
	assert.NoError(t, err)

	err = taskDao.CancelTasks(ctx, mode, []string{taskInMgo.TaskID})
	assert.NoError(t, err)

	_, err = taskDao.QueryByID(ctx, mode, taskId)
	assert.NoError(t, err)

	err = taskDao.Remove(ctx, mode, taskId)
	assert.NoError(t, err)

	err = jobDao.Remove(ctx, jobId)
	assert.NoError(t, err)
}

func TestQueryResult(t *testing.T) {
	var (
		ctx     = context.Background()
		mgoConf = CapMgoConfig{
			IdleTimeout:  5000000000,
			MgoPoolLimit: 5,
			Mgo: mgoutil.Config{
				Host:           "127.0.0.1:27017",
				DB:             "argus_cap_test",
				Mode:           "strong",
				SyncTimeoutInS: 5,
			},
		}
	)

	taskDao, err := NewTaskDao(&mgoConf)
	assert.NoError(t, err)

	var (
		mode    = "mode_politician_pulp_terror"
		jobId   = "1ccfdac8-e97c-4242-9964-b86b1deb8514_image"
		bufLine = 0
		tasks   = []TaskInMgo{TaskInMgo{
			TaskID: "0",
			JobID:  jobId,
			URI:    "http://testUrl.com",
			Status: string(enums.TaskDoing),
		}, TaskInMgo{
			TaskID: "1",
			JobID:  jobId,
			URI:    "http://testUrl.com",
			Status: string(enums.TaskDoing),
		}}
	)

	err = taskDao.Insert(ctx, mode, tasks...)
	assert.NoError(t, err)
	marker := ""
	for {
		var getTasks []TaskInMgo
		marker, getTasks, err = taskDao.QueryResults(ctx, mode, jobId, marker, 100)
		assert.NoError(t, err)
		bufLine += len(getTasks)
		if len(getTasks) == 0 {
			break
		}
	}
	assert.Equal(t, 2, bufLine)

	for _, v := range tasks {
		err = taskDao.Remove(ctx, mode, v.TaskID)
		assert.NoError(t, err)
	}
}
