package db

import (
	"context"
	"testing"

	"gopkg.in/mgo.v2/bson"

	"github.com/qiniu/db/mgoutil.v3"
)

func init() {
	Init(&mgoutil.Config{
		Host: "mongodb://127.0.0.1:27017",
		DB:   "argus_test",
	})
}

func TestFindFace(t *testing.T) {
	faceDao, err := NewFaceDao()
	if err != nil {
		t.Fatal("new face dao error", err)
	}
	id := bson.NewObjectId()

	err = faceDao.Insert(
		context.Background(),
		Face{
			ID:        id,
			ClusterID: 1,
			Feature:   FaceFeature{Feature: "test"},
			GtID:      2,
		})

	if err != nil {
		t.Error("insert error", err)
		return
	}
	defer faceDao.Remove(id)
	features, err := faceDao.Find(id)
	if err != nil {
		t.Error("find feature error", err)
		return
	}

	if len(features) != 1 {
		t.Errorf("feature error:%v", features)
		return
	}

	if features[0].ClusterID != 1 || features[0].GtID != 2 {
		t.Error("feature content error")
	}
}
