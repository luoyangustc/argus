package db

import (
	"context"
	"testing"
	"time"

	"github.com/qiniu/db/mgoutil.v3"
)

func init() {
	Init(&mgoutil.Config{
		Host: "mongodb://127.0.0.1:27017",
		DB:   "argus_test",
	})
}

func TestClusterTask(t *testing.T) {
	clusterDao, err := NewClusterTaskDao()

	if err != nil {
		t.Fatal("new dao error", err)
	}

	{
		task := ClusterTask{UID: "uid", Euid: "euid", CreatedAt: time.Now()}
		if err = clusterDao.UpsertTask(context.Background(), task); err != nil {
			t.Fatal("upsert data error", err)
		}
	}
	{
		task := ClusterTask{UID: "uid", Euid: "euid", CreatedAt: time.Now()}
		if err = clusterDao.UpsertTask(context.Background(), task); err != nil {
			t.Fatal("upsert data error", err)
		}
	}
	{
		task := ClusterTask{UID: "uid", Euid: "euid1", CreatedAt: time.Now()}
		if err = clusterDao.UpsertTask(context.Background(), task); err != nil {
			t.Fatal("upsert data error", err)
		}
	}

	{
		task, err := clusterDao.FindTask(context.Background())
		if err != nil || task == nil {
			t.Fatal("find task failed", err, task)
		}
		if err = clusterDao.DoneTask(context.Background(), *task); err != nil {
			t.Fatal("done task failed", err)
		}
	}
	{
		task, err := clusterDao.FindTask(context.Background())
		if err != nil || task == nil {
			t.Fatal("find task failed", err, task)
		}
		if err = clusterDao.DoneTask(context.Background(), *task); err != nil {
			t.Fatal("done task failed", err)
		}
	}
}
