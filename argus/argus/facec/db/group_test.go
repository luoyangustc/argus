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

func TestFindGroup(t *testing.T) {
	groupDao, err := NewGroupDao()
	if err != nil {
		t.Fatal("new group dao error")
	}

	defer groupDao.RemoveAll("uid", "euid")
	defer groupDao.RemoveAll("", "euid")

	err = groupDao.Insert(context.Background(),
		[]GroupInfo{
			{
				ID:   bson.NewObjectId(),
				Euid: "euid",
				UID:  "uid",
				Group: Group{
					ID:        1,
					FaceCount: 100,
					Files:     []string{"f1", "f2"},
					Refs: []Ref{
						{
							File: "f1",
							Pts: [][]int64{
								{1, 2}, {3, 4}, {5, 6}, {7, 8},
							},
							Score: 1.2,
						},
					},
				},
			},
			{
				ID:   bson.NewObjectId(),
				Euid: "euid",
				Group: Group{
					ID:        2,
					FaceCount: 100,
					Files:     []string{"f1", "f2"},
					Refs: []Ref{
						{
							File: "f1",
							Pts: [][]int64{
								{1, 2}, {3, 4}, {5, 6}, {7, 8},
							},
							Score: 1.2,
						},
					},
				},
			},
			{
				ID:   bson.NewObjectId(),
				Euid: "euid",
				UID:  "uid",
				Group: Group{
					ID:        3,
					FaceCount: 100,
					Files:     []string{"f1", "f2"},
					Refs: []Ref{
						{
							File: "f1",
							Pts: [][]int64{
								{1, 2}, {3, 4}, {5, 6}, {7, 8},
							},
							Score: 1.2,
						},
					},
				},
			},
			{
				ID:   bson.NewObjectId(),
				Euid: "euid",
				UID:  "uid",
				Group: Group{
					ID:        2,
					FaceCount: 100,
					Files:     []string{"f1", "f2"},
					Refs: []Ref{
						{
							File: "f1",
							Pts: [][]int64{
								{1, 2}, {3, 4}, {5, 6}, {7, 8},
							},
							Score: 1.2,
						},
					},
					Faces: []FaceG{
						{
							File: "f1",
							Pts: [][]int64{
								{1, 2}, {3, 4}, {5, 6}, {7, 8},
							},
							Score:             1.2,
							ClusterCenterDist: 0.9,
						},
					},
				},
			},
		})

	if err != nil {
		t.Error("insert error", err)
		return
	}

	found := func(uid, euid string, groupID int64) bool {
		group, err := groupDao.FindGroup(uid, euid, groupID)
		if err != nil {
			t.Error("find group error", err)
			return false
		}

		if group.Group.ID != groupID {
			t.Error("group id error")
			return false
		}
		return true
	}

	t.Run("FindGroup", func(t *testing.T) {
		if !found("uid", "euid", 1) {
			return
		}
		if !found("", "euid", 2) {
			return
		}

		_, err = groupDao.FindGroup("uid", "euid", 2)
		if err != nil {
			t.Error("find group error", err)
			return
		}

		groups, err := groupDao.FindGroups("uid", "euid", 0, 100)
		if err != nil {
			t.Error("find groups error", err)
			return
		}

		if len(groups) != 3 {
			t.Error("group length error", len(groups))
			return
		}

		groups, _ = groupDao.FindGroups("uid", "euid", 1, 100)
		if len(groups) != 2 {
			t.Error("group length error", len(groups))
			return
		}

		if groups[0].Group.ID != 2 {
			t.Error("group id error", groups[0].Group.ID)
			return
		}
	})

	t.Run("FindSimpleGroups", func(t *testing.T) {
		groups, err := groupDao.FindSimpleGroups("uid", "euid", 0, 100)

		if err != nil {
			t.Error("find groups error", err)
			return
		}

		if len(groups) != 3 {
			t.Error("group count error", len(groups))
			return
		}

		if len(groups[0].Group.Faces) != 0 {
			t.Error("face count error", len(groups[0].Group.Faces))
			return
		}

		if len(groups[1].Group.Faces) != 0 {
			t.Error("face count error", len(groups[1].Group.Faces))
			return
		}
	})

	t.Run("Remove", func(t *testing.T) {
		err = groupDao.Remove("uid", "euid", 1)
		if err != nil {
			t.Error("remove group error")
			return
		}

		_, err = groupDao.FindGroup("uid", "euid", 1)
		if err == nil {
			t.Error("DO not remove group")
		}
	})
}

func TestAppendImage(t *testing.T) {
	// TODO
}

func TestMoveImage(t *testing.T) {
	// TODO
}
