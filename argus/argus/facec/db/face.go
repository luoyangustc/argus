package db

import (
	"time"

	"context"

	"gopkg.in/mgo.v2/bson"

	"github.com/qiniu/db/mgoutil.v3"
	"github.com/qiniu/xlog.v1"
)

// Face maping the collection faces
type Face struct {
	ID                bson.ObjectId `bson:"_id,omitempty"`
	UID               string        `bson:"uid"`
	Euid              string        `bson:"euid"`
	File              string        `bson:"file"`
	ClusterID         int64         `bson:"cluster_id"`
	GtID              int64         `bson:"gt_id"`
	Score             float64       `bson:"score"`
	ClusterCenterDist float64       `bson:"cluster_center_dist"`
	CreatedAt         time.Time     `bson:"created_at"`
	Pts               FacePts       `bson:"pts"`
	Feature           FaceFeature   `bson:"feature"`
}

type FacePts struct {
	Det    [][]int64 `bson:"det"`
	ModelV string    `bson:"modelv"`
}

type FaceFeature struct {
	Feature string `bson:"feature"`
	ModelV  string `bson:"modelv"`
}

//const faceCollection = "faces"

// FaceDao faces collection dao
type FaceDao interface {
	FindByEuid(ctx context.Context, uid, euid string) ([]Face, error)
	FindByEuidWithoutFeature(ctx context.Context, uid, euid string) ([]Face, error)
	Insert(ctx context.Context, faces ...Face) error
	UpdateFeatures(ctx context.Context, faces []Face) (int, error)
	UpdateClusterAndCenterDist(ctx context.Context, faces []Face) error
	UpdateGtId(ctx context.Context, togroup int64, faces ...bson.ObjectId) error

	// Deprecated
	Find(faceIDs ...bson.ObjectId) ([]Face, error)
	FindWithOutFeature(faceIDs ...bson.ObjectId) ([]Face, error)
	Remove(faceIDs ...bson.ObjectId) error
	MaxScoreRefs(uid, euid string, groupID int64) (*Ref, error)
}

// NewFaceDao new face dao
func NewFaceDao() (FaceDao, error) {
	return &faceDao{coll: &collections.Faces}, nil
}

type faceDao struct{ coll *mgoutil.Collection }

func (d *faceDao) EnsureIndexes() error {
	return nil
}

func (d *faceDao) FindByEuid(ctx context.Context, uid, euid string) ([]Face, error) {
	c := d.coll.CopySession()
	defer c.CloseSession()

	xl := xlog.FromContextSafe(ctx)

	var faces []Face
	err := c.Find(bson.M{"uid": uid, "euid": euid}).All(&faces)
	if err != nil {
		xl.Error("query user's faces error", err)
		return nil, err
	}
	return faces, nil
}

func (d *faceDao) FindByEuidWithoutFeature(ctx context.Context, uid, euid string) ([]Face, error) {
	c := d.coll.CopySession()
	defer c.CloseSession()

	xl := xlog.FromContextSafe(ctx)

	var faces []Face
	err := c.Find(bson.M{"uid": uid, "euid": euid}).Select(bson.M{"feature": 0}).All(&faces)
	if err != nil {
		xl.Error("query user's faces error", err)
		return nil, err
	}
	return faces, nil
}

func (d *faceDao) Insert(ctx context.Context, faces ...Face) error {
	if len(faces) <= 0 {
		return nil
	}

	c := d.coll.CopySession()
	defer c.CloseSession()

	xl := xlog.FromContextSafe(ctx)

	var insertFaces []interface{} = make([]interface{}, 0, len(faces))
	for i, len := 0, len(faces); i < len; i++ {
		insertFaces = append(insertFaces, &faces[i])
	}

	err := c.Insert(insertFaces...)
	if err != nil {
		xl.Errorf("insert face error: %d %v", len(faces), err)
	}
	return err
}

func (d *faceDao) UpdateFeatures(ctx context.Context, faces []Face) (int, error) {
	c := d.coll.CopySession()
	defer c.CloseSession()

	xl := xlog.FromContextSafe(ctx)

	xl.Debugf("update features:%v", faces)

	var updateParams []interface{} = make([]interface{}, 0, len(faces))
	for _, f := range faces {
		updateParams = append(updateParams,
			bson.M{
				"_id": f.ID,
			},
			bson.M{
				"$set": bson.M{"feature": f.Feature},
			},
		)
	}
	bulk := c.Bulk()
	bulk.Update(updateParams...)
	ch, err := bulk.Run()

	if err != nil {
		xl.Error("update error", err)
		return 0, err
	}
	return ch.Modified, nil
}

func (d *faceDao) UpdateClusterAndCenterDist(ctx context.Context, faces []Face) error {
	c := d.coll.CopySession()
	defer c.CloseSession()

	xl := xlog.FromContextSafe(ctx)

	var updateParams []interface{}
	for _, f := range faces {
		updateParams = append(updateParams,
			bson.M{
				"_id": f.ID,
			},
			bson.M{
				"$set": bson.M{
					"cluster_id":          f.ClusterID,
					"cluster_center_dist": f.ClusterCenterDist,
				},
			},
		)
	}
	bulk := c.Bulk()
	bulk.Update(updateParams...)
	_, err := bulk.Run()

	if err != nil {
		xl.Error("update error", err)
	}
	return err
}

func (d *faceDao) UpdateGtId(ctx context.Context, togroup int64, faces ...bson.ObjectId) error {
	c := d.coll.CopySession()
	defer c.CloseSession()

	xl := xlog.FromContextSafe(ctx)

	_, err := c.UpdateAll(
		bson.M{"_id": bson.M{"$in": faces}},
		bson.M{"$set": bson.M{"gt_id": togroup}},
	)
	if err != nil {
		xl.Error("update gt id error", err)
	}
	return err
}

////////////////////////////////////////////////////////////////////////////////
// Deprecated

func (d *faceDao) Find(faceIDs ...bson.ObjectId) ([]Face, error) {
	c := d.coll.CopySession()
	defer c.CloseSession()

	var faces []Face
	err := c.Find(bson.M{"_id": bson.M{"$in": faceIDs}}).All(&faces)
	if err != nil {
		xlog.Errorf("", "query user's faces error", err)
		return nil, err

	}

	return faces, nil
}

func (d *faceDao) FindWithOutFeature(faceIDs ...bson.ObjectId) ([]Face, error) {
	c := d.coll.CopySession()
	defer c.CloseSession()

	var faces []Face
	err := c.Find(bson.M{"_id": bson.M{"$in": faceIDs}}).Select(bson.M{"feature": 0}).All(&faces)
	if err != nil {
		xlog.Error("query user's faces error", err)
		return nil, err
	}

	return faces, nil
}

func (d *faceDao) Remove(faceIDs ...bson.ObjectId) error {
	c := d.coll.CopySession()
	defer c.CloseSession()

	_, err := c.RemoveAll(bson.M{"_id": bson.M{"$in": faceIDs}})
	if err != nil {
		xlog.Errorf("", "remove face error", err)
	}
	return err
}

func (d *faceDao) MaxScoreRefs(uid, euid string, groupID int64) (*Ref, error) {

	c := d.coll.CopySession()
	defer c.CloseSession()

	var r []Ref
	err := c.Find(bson.M{
		"uid":   uid,
		"euid":  euid,
		"gt_id": groupID,
	}).Sort("-score").Limit(1).Select(bson.M{"_id": 0, "score": 1, "pts": 1, "file": 1}).All(&r)

	if err != nil {
		xlog.Errorf("", "get refs from face table error:%v", err)
		return nil, err
	}
	return &r[0], nil
}
