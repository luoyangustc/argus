package dao

import (
	"context"
	"time"

	"gopkg.in/mgo.v2"
	"gopkg.in/mgo.v2/bson"

	"github.com/qiniu/db/mgoutil.v3"
	"github.com/qiniu/xlog.v1"
)

type IGroupDAO interface {
	QueryByGID(ctx context.Context, groupID string) (*GroupInMgo, error)
	QueryAll(ctx context.Context) ([]*GroupInMgo, error)
	Insert(ctx context.Context, groups ...GroupInMgo) error
	Update(ctx context.Context, group *GroupInMgo) error
	Remove(ctx context.Context, groupID string) error
}

////////////////////////////////////////////////////////////////////////////////

var _ IGroupDAO = _GroupInMgo{}

type _GroupInMgo struct {
	coll *mgoutil.Collection
}

// NewGroupInMgo New
func NewGroupInMgo(coll *mgoutil.Collection) IGroupDAO {

	coll.EnsureIndex(mgo.Index{Key: []string{"group_id"}, Unique: true})

	return _GroupInMgo{coll: coll}
}

func (m _GroupInMgo) QueryByGID(ctx context.Context, groupID string) (*GroupInMgo, error) {
	coll := m.coll.CopySession()
	defer coll.CloseSession()

	var (
		xl    = xlog.FromContextSafe(ctx)
		group GroupInMgo
	)

	err := coll.Find(bson.M{"group_id": groupID}).One(&group)
	if err != nil {
		xl.Warnf("find group failed. %s %v", groupID, err)
		return nil, err
	}

	return &group, nil
}

func (m _GroupInMgo) QueryAll(ctx context.Context) ([]*GroupInMgo, error) {
	coll := m.coll.CopySession()
	defer coll.CloseSession()

	var (
		xl     = xlog.FromContextSafe(ctx)
		groups = []*GroupInMgo{}
	)

	err := coll.Find(bson.M{}).All(&groups)
	if err != nil {
		xl.Warnf("find group failed. %v", err)
	}

	return groups, err
}

func (m _GroupInMgo) Insert(ctx context.Context, groups ...GroupInMgo) error {
	coll := m.coll.CopySession()
	defer coll.CloseSession()

	for _, group := range groups {

		group.ID = bson.NewObjectId()
		group.CreatedAt = time.Now()
		group.UpdatedAt = group.CreatedAt
		group.Version = 1

		err := coll.Insert(group)
		if err != nil {
			return err
		}
	}

	return nil
}

func (m _GroupInMgo) Update(ctx context.Context, group *GroupInMgo) error {
	coll := m.coll.CopySession()
	defer coll.CloseSession()

	group.UpdatedAt = time.Now()
	version := group.Version
	group.Version++

	return coll.Update(bson.M{"_id": group.ID, "version": version}, group)
}

func (m _GroupInMgo) Remove(ctx context.Context, groupID string) error {
	coll := m.coll.CopySession()
	defer coll.CloseSession()

	return coll.Remove(bson.M{"group_id": groupID})
}
