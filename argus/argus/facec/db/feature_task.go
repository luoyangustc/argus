package db

import (
	"context"
	"time"

	"gopkg.in/mgo.v2"
	"gopkg.in/mgo.v2/bson"

	"github.com/qiniu/db/mgoutil.v3"
	"github.com/qiniu/xlog.v1"

	"qiniu.com/argus/argus/facec/dbbase"
)

type FeatureTask struct {
	ID           bson.ObjectId `bson:"_id,omitempty"`
	UID          string        `bson:"uid"`    // 可以去除
	Euid         string        `bson:"euid"`   // 可以去除
	Status       Status        `bson:"status"` // 0: todo; 1: doing
	CreatedAt    time.Time     `bson:"created_at"`
	LastModified time.Time     `bson:"last_modified"`
	Face         struct {
		ID   bson.ObjectId `bson:"id"`
		File string        `bson:"file"`
		Pts  FacePts       `bson:"pts"`
	} `bson:"face"`
}

type FeatureTaskDao interface {
	FindTasks(ctx context.Context, count int) ([]FeatureTask, error)
	Insert(ctx context.Context, task ...FeatureTask) error
	Remove(ctx context.Context, ids ...bson.ObjectId) error
}

func NewFeatureTaskDao() (FeatureTaskDao, error) {
	return &featureTaskDao{coll: &collections.FaceFeatureTasks}, nil
}

type featureTaskDao struct{ coll *mgoutil.Collection }

func (d *featureTaskDao) EnsureIndexes() error {
	coll := d.coll.CopySession()
	defer coll.CloseSession()

	if err := coll.EnsureIndex(mgo.Index{Key: []string{"uid", "euid", "face.id"}, Unique: true}); err != nil {
		return err
	}
	if err := coll.EnsureIndex(mgo.Index{Key: []string{"created_at"}}); err != nil {
		return err
	}
	if err := coll.EnsureIndex(mgo.Index{Key: []string{"status"}}); err != nil {
		return err
	}
	return nil
}

func (d *featureTaskDao) FindTasks(ctx context.Context, count int) ([]FeatureTask, error) {
	coll := d.coll.CopySession()
	defer coll.CloseSession()

	xl := xlog.FromContextSafe(ctx)

	if count > dbbase.MaxLimit {
		count = dbbase.MaxLimit
	}

	change := mgo.Change{
		Update:    bson.M{"$set": bson.M{"status": STATUS_DOING, "last_modified": time.Now()}},
		ReturnNew: true,
	}

	var tasks []FeatureTask = make([]FeatureTask, count)
	for i := 0; i < count; i++ {
		_, err := coll.Find(bson.M{"status": STATUS_TODO}).Sort("created_at").Apply(change, &tasks[i])
		if err == mgo.ErrNotFound {
			tasks = tasks[:i]
			break
		}
		if err != nil {
			xl.Error("find new feature tasks error", len(tasks[:i]), err)
			return tasks[:i], err
		}
	}
	xl.Infof("find new feature tasks. %d", len(tasks))
	return tasks, nil
}

func (d *featureTaskDao) Insert(ctx context.Context, tasks ...FeatureTask) error {
	if len(tasks) <= 0 {
		return nil
	}

	xl := xlog.FromContextSafe(ctx)

	coll := d.coll.CopySession()
	defer coll.CloseSession()

	var _tasks []interface{} = make([]interface{}, 0, len(tasks))
	for i, n := 0, len(tasks); i < n; i++ {
		_tasks = append(_tasks, &tasks[i])
	}

	err := coll.Insert(_tasks...)
	if err != nil {
		xl.Error("insert the feature task error:", err)
	}
	return err
}

func (d *featureTaskDao) Remove(ctx context.Context, ids ...bson.ObjectId) error {
	if len(ids) <= 0 {
		return nil
	}

	xl := xlog.FromContextSafe(ctx)

	coll := d.coll.CopySession()
	defer coll.CloseSession()

	info, err := coll.RemoveAll(bson.M{"_id": bson.M{"$in": ids}})
	if err != nil {
		xl.Error("remove the feature task error:", err)
		return err
	}
	xl.Infof("remove the feature task: expected %d , matched %d , remove %d", len(ids), info.Matched, info.Removed)
	return nil
}
