package db

import (
	"context"

	"gopkg.in/mgo.v2"
	"gopkg.in/mgo.v2/bson"

	"github.com/qiniu/db/mgoutil.v3"
	"github.com/qiniu/xlog.v1"
)

// DataVersion store the data version information
type DataVersion struct {
	ID    bson.ObjectId `bson:"_id"`
	UID   string        `bson:"uid"`
	Euid  string        `bson:"euid"`
	Group GroupV        `bson:"final_group"`
}

// GroupV store the group data version
type GroupV struct {
	Version string `bson:"version"`
	Status  Status `bson:"status"`
}

// DataVersionDao collection operations
type DataVersionDao interface {
	FindGroupVersion(ctx context.Context, uid, euid string) (*GroupV, error)
	UpdateStatus(ctx context.Context, uid, euid string, oldStatus, newStatus Status) (*GroupV, error)
	UpdateStatusAndVersion(
		ctx context.Context,
		uid, euid string,
		oldVersion, newVersion string,
		oldStatus, newStatus Status,
	) (bool, error)
}

// NewDataVersionDao create dao
func NewDataVersionDao() (DataVersionDao, error) {
	return &dataVersionDao{coll: &collections.DataVersions}, nil
}

type dataVersionDao struct {
	coll *mgoutil.Collection
}

func (d *dataVersionDao) FindGroupVersion(ctx context.Context, uid, euid string) (*GroupV, error) {
	c := d.coll.CopySession()
	defer c.CloseSession()

	var dv DataVersion
	err := c.Find(bson.M{"uid": uid, "euid": euid}).One(&dv)
	if err != nil {
		return nil, err
	}
	return &dv.Group, nil
}

func (d *dataVersionDao) UpdateStatus(
	ctx context.Context,
	uid, euid string,
	oldStatus, newStatus Status,
) (*GroupV, error) {
	c := d.coll.CopySession()
	defer c.CloseSession()

	xl := xlog.FromContextSafe(ctx)

	var dv DataVersion
	ch, err := c.Find(bson.M{
		"uid":  uid,
		"euid": euid,
		"$or": []bson.M{
			bson.M{"final_group.status": bson.M{"$exists": false}},
			bson.M{"final_group.status": oldStatus},
		},
	}).Apply(mgo.Change{
		Update: bson.M{
			"$set": bson.M{
				"final_group.status": newStatus,
			},
		},
		Upsert:    true,
		ReturnNew: true,
	}, &dv)

	if err != nil {
		xl.Errorf("update status error %v", err)
		return nil, err
	}
	xl.Infof("changed %d, upsertid:%v", ch.Updated, ch.UpsertedId)

	return &dv.Group, nil
}

func (d *dataVersionDao) UpdateStatusAndVersion(
	ctx context.Context,
	uid, euid string,
	oldVersion, newVersion string,
	oldStatus, newStatus Status,
) (bool, error) {
	c := d.coll.CopySession()
	defer c.CloseSession()

	xl := xlog.FromContextSafe(ctx)

	var dv DataVersion
	ch, err := c.Find(bson.M{
		"uid":  uid,
		"euid": euid,
		"$and": []interface{}{
			bson.M{
				"$or": []bson.M{
					bson.M{"final_group.status": bson.M{"$exists": false}},
					bson.M{"final_group.status": oldStatus},
				},
			},
			bson.M{
				"$or": []bson.M{
					bson.M{"final_group.version": bson.M{"$exists": false}},
					bson.M{"final_group.version": oldVersion},
				},
			},
		},
	}).Apply(mgo.Change{
		Update: bson.M{
			"$set": bson.M{
				"final_group.status":  newStatus,
				"final_group.version": newVersion,
			},
		},
	}, &dv)

	if err != nil {
		xl.Errorf("update done error: %v", err)
		return false, err
	}
	xl.Infof("changed %d, upsertid:%v", ch.Updated, ch.UpsertedId)

	return dv.Group.Status == oldStatus && dv.Group.Version == oldVersion, nil
}
