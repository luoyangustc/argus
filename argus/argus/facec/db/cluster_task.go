package db

import (
	"time"

	"context"

	"gopkg.in/mgo.v2"
	"gopkg.in/mgo.v2/bson"

	"github.com/qiniu/db/mgoutil.v3"
	"github.com/qiniu/xlog.v1"

	"qiniu.com/argus/argus/facec/dbbase"
)

//const faceClusterTaskCollection = "face_cluster_tasks"

// ClusterTask mapping collection face_cluster_task
type ClusterTask struct {
	ID           bson.ObjectId `bson:"_id,omitempty"`
	UID          string        `bson:"uid"`
	Euid         string        `bson:"euid"`
	Status       Status        `bson:"status"` // 0: todo; 1: doing
	CreatedAt    time.Time     `bson:"created_at"`
	LastModified time.Time     `bson:"last_modified"`

	File string `bson:"file"` // Deprecated
}

// Deprecated
// NewImages contains the images to extra face
type NewImages struct {
	User  User            `bson:"_id"`
	Files []string        `bson:"files"`
	IDs   []bson.ObjectId `bson:"ids"`
}

// Deprecated
// User ideneity
type User struct {
	UID  string
	Euid string
}

// ClusterTaskDao interface
type ClusterTaskDao interface {
	FindTask(ctx context.Context) (*ClusterTask, error)
	UpsertTask(ctx context.Context, task ClusterTask) error
	DoneTask(ctx context.Context, task ClusterTask) error

	FindNewImages(count int) ([]NewImages, error) // Deprecated
	Insert(task ...ClusterTask) error             // Deprecated
	Remove(ids ...bson.ObjectId) error            // Deprecated
	RemoveByUser(users ...User) error             // Deprecated
}

// NewClusterTaskDao new cluster task dao
func NewClusterTaskDao() (ClusterTaskDao, error) {
	return &clusterTaskDao{coll: &collections.FaceClusterTasks}, nil
}

type clusterTaskDao struct{ coll *mgoutil.Collection }

func (d *clusterTaskDao) EnsureIndexes() error {
	coll := d.coll.CopySession()
	defer coll.CloseSession()

	if err := coll.EnsureIndex(mgo.Index{Key: []string{"uid", "euid"}, Unique: true}); err != nil {
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

func (d *clusterTaskDao) FindTask(ctx context.Context) (*ClusterTask, error) {
	coll := d.coll.CopySession()
	defer coll.CloseSession()

	xl := xlog.FromContextSafe(ctx)

	change := mgo.Change{
		Update:    bson.M{"$set": bson.M{"status": STATUS_DOING, "last_modified": time.Now()}},
		ReturnNew: true,
	}

	var task ClusterTask
	_, err := coll.
		Find(bson.M{
			"$or": []bson.M{
				bson.M{"status": bson.M{"$exists": false}},
				bson.M{"status": STATUS_TODO}},
		}).
		Sort("created_at").
		Apply(change, &task)
	if err == mgo.ErrNotFound {
		return nil, nil
	}
	if err != nil {
		xl.Error("find new feature tasks error", err)
		return nil, err
	}
	xl.Infof("find new feature tasks. %#v", task)
	return &task, nil
}

func (d *clusterTaskDao) UpsertTask(ctx context.Context, task ClusterTask) error {
	coll := d.coll.CopySession()
	defer coll.CloseSession()

	//xl := xlog.FromContextSafe(ctx)
	change := mgo.Change{
		Update: bson.M{"$set": bson.M{"created_at": task.CreatedAt}},
		Upsert: true,
	}
	_, err := coll.Find(
		bson.M{
			"uid":        task.UID,
			"euid":       task.Euid,
			"created_at": bson.M{"$lt": task.CreatedAt},
		}).
		Apply(change, nil)
	if err != nil && mgo.IsDup(err) {
		err = nil
	}
	return err
}

func (d *clusterTaskDao) DoneTask(ctx context.Context, task ClusterTask) error {
	coll := d.coll.CopySession()
	defer coll.CloseSession()

	xl := xlog.FromContextSafe(ctx)
	err := coll.Remove(bson.M{"_id": task.ID, "created_at": task.CreatedAt})
	if err == nil {
		xl.Infof("remove the cluster task: %#v", task)
		return nil
	}
	if err != mgo.ErrNotFound {
		xl.Errorf("remove the cluster task: %#v %s", task, err)
		return err
	}
	change := mgo.Change{
		Update: bson.M{"$set": bson.M{"status": STATUS_TODO, "last_modified": time.Now()}},
	}
	_, err = coll.Find(bson.M{"_id": task.ID}).Apply(change, nil)
	if err == nil {
		xl.Infof("done the cluster task: %#v", task)
	} else {
		xl.Errorf("done the cluster task: %#v %s", task, err)
	}
	return err
}

////////////////////////////////////////////////////////////////////////////////
// Deprecated
func (d *clusterTaskDao) FindNewImages(count int) ([]NewImages, error) {
	c := d.coll.CopySession()
	defer c.CloseSession()

	if count > dbbase.MaxLimit {
		count = dbbase.MaxLimit
	}

	var images []NewImages
	err := c.Pipe([]bson.M{
		bson.M{
			"$group": bson.M{
				"_id":        bson.M{"euid": "$euid", "uid": "$uid"},
				"files":      bson.M{"$push": "$file"},
				"ids":        bson.M{"$push": "$_id"},
				"created_at": bson.M{"$min": "$created_at"},
			},
		},
		bson.M{
			"$sort": bson.M{"created_at": 1},
		},
		bson.M{
			"$limit": count,
		},
	}).All(&images)

	if err != nil {
		xlog.Errorf("", "find new image error", err)
	}
	return images, err
}

// Deprecated
func (d *clusterTaskDao) Insert(images ...ClusterTask) error {
	if len(images) <= 0 {
		return nil
	}

	c := d.coll.CopySession()
	defer c.CloseSession()

	var insertImages []interface{}
	for i, len := 0, len(images); i < len; i++ {
		insertImages = append(insertImages, &images[i])
	}
	err := c.Insert(insertImages...)
	if err != nil {
		xlog.Errorf("", "insert the cluster task error:", err)
	}
	return err
}

// Deprecated
func (d *clusterTaskDao) Remove(ids ...bson.ObjectId) error {
	if len(ids) <= 0 {
		return nil
	}

	c := d.coll.CopySession()
	defer c.CloseSession()

	info, err := c.RemoveAll(bson.M{"_id": bson.M{"$in": ids}})
	if err != nil {
		xlog.Errorf("", "remove the cluster task error:", err)
		return err
	}

	xlog.Debugf("", "remove %d, expected %d", info.Removed, len(ids))

	return nil
}

// Deprecated
func (d *clusterTaskDao) RemoveByUser(users ...User) error {
	c := d.coll.CopySession()
	defer c.CloseSession()

	var conds []bson.M
	for _, u := range users {
		conds = append(conds, bson.M{"uid": u.UID, "euid": u.Euid})
	}
	_, err := c.RemoveAll(bson.M{"$or": conds})

	if err != nil {
		xlog.Errorf("", "remove error")
	}
	return err
}
