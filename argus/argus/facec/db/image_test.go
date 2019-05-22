package db

import (
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

func TestFindImages(t *testing.T) {
	imageDao, err := NewImageDao()
	defer imageDao.Remove("uid", "euid")
	defer imageDao.Remove("", "euid")

	if err != nil {
		t.Fatal("new image dao error", err)
	}

	err = imageDao.Insert([]Image{
		{
			ID:    bson.NewObjectId(),
			Euid:  "euid",
			UID:   "uid",
			Faces: []bson.ObjectId{bson.NewObjectId(), bson.NewObjectId()},
			File:  "file",
		},
		{
			ID:    bson.NewObjectId(),
			Euid:  "euid",
			Faces: []bson.ObjectId{bson.NewObjectId(), bson.NewObjectId()},
			File:  "file2",
		},
	})

	if err != nil {
		t.Error("insert error", err)
		return
	}

	images, err := imageDao.FindImages("uid", "euid")
	if len(images) != 1 {
		t.Errorf("find image count error:exp/act,%d/%d, %v", 1, len(images), images)
		return
	}

	images, err = imageDao.FindImages("", "euid")
	if len(images) != 1 {
		t.Errorf("find image count error:exp/act,%d/%d", 1, len(images))
		return
	}

	images, err = imageDao.FindImageByFiles("uid", "euid", []string{"1"})
	if len(images) != 0 {
		t.Errorf("find image count error:exp/act,%d/%d, %v", 0, len(images), images)
		return
	}

	images, err = imageDao.FindImageByFiles("uid", "euid", []string{})
	if len(images) != 0 {
		t.Errorf("find image count error:exp/act,%d/%d, %v", 0, len(images), images)
		return
	}

	err = imageDao.Insert([]Image{
		{
			ID:    bson.NewObjectId(),
			Euid:  "euid",
			UID:   "uid",
			Faces: []bson.ObjectId{bson.NewObjectId(), bson.NewObjectId()},
			File:  "file222",
		},
		{
			ID:    bson.NewObjectId(),
			Euid:  "euid",
			UID:   "uid",
			Faces: []bson.ObjectId{bson.NewObjectId(), bson.NewObjectId()},
			File:  "file333",
		},
	})

	if err != nil {
		t.Error("insert error", err)
		return
	}

	images, err = imageDao.FindImageByFiles("uid", "euid", []string{"file"})
	if len(images) != 1 {
		t.Errorf("find image count error:exp/act,%d/%d, %v", 1, len(images), images)
		return
	}

	images, err = imageDao.FindImageByFiles("uid", "euid", []string{"file", "file333"})
	if len(images) != 2 {
		t.Errorf("find image count error:exp/act,%d/%d, %v", 2, len(images), images)
		return
	}
}

func TestFindFaces(t *testing.T) {
	imageDao, err := NewImageDao()
	defer imageDao.Remove("uid", "euid")
	err = imageDao.Insert([]Image{
		{
			ID:    bson.NewObjectId(),
			Euid:  "euid",
			UID:   "uid",
			Faces: []bson.ObjectId{bson.NewObjectId(), bson.NewObjectId()},
			File:  "file",
		},
		{
			ID:    bson.NewObjectId(),
			Euid:  "euid",
			UID:   "uid",
			Faces: []bson.ObjectId{bson.NewObjectId()},
			File:  "file1",
		},
	})

	if err != nil {
		t.Error("insert error", err)
		return
	}

	faces, err := imageDao.FindFaces("uid", "euid")
	if err != nil {
		t.Error("find face error", err)
		return
	}

	if len(faces) != 2 {
		t.Error("face count error, exp:2, act:", len(faces), faces)
		return
	}

	faces, err = imageDao.FindFaces("uid", "euid", "file")
	if err != nil {
		t.Error("find face error", err)
		return
	}

	if len(faces) != 1 {
		t.Error("feature count error, exp:3, act:", len(faces), faces)
		return
	}
}
