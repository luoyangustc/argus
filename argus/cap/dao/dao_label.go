package dao

import (
	"context"
	"time"

	"gopkg.in/mgo.v2"
	"gopkg.in/mgo.v2/bson"

	"github.com/qiniu/db/mgoutil.v3"
	"github.com/qiniu/xlog.v1"
)

type ILabelDAO interface {
	QueryByName(ctx context.Context, name string) (*LabelInMgo, error)
	QueryAll(ctx context.Context) ([]*LabelInMgo, error)
	Insert(ctx context.Context, labels ...LabelInMgo) error
	Update(ctx context.Context, label *LabelInMgo) error
	Remove(ctx context.Context, name string) error
}

////////////////////////////////////////////////////////////////////////////////

var _ ILabelDAO = _LabelDAO{}

type _LabelDAO struct {
	coll *mgoutil.Collection
}

// NewLabelInMgo New
func NewLabelInMgo(coll *mgoutil.Collection) _LabelDAO {

	coll.EnsureIndex(mgo.Index{Key: []string{"name"}, Unique: true})

	return _LabelDAO{coll: coll}
}

func (m _LabelDAO) QueryByName(ctx context.Context, name string) (*LabelInMgo, error) {
	coll := m.coll.CopySession()
	defer coll.CloseSession()

	var (
		xl    = xlog.FromContextSafe(ctx)
		label LabelInMgo
	)

	err := coll.Find(bson.M{"name": name}).One(&label)
	if err != nil {
		xl.Warnf("find label failed. %s %v", name, err)
		return nil, err
	}

	return &label, nil
}

func (m _LabelDAO) QueryAll(ctx context.Context) ([]*LabelInMgo, error) {
	coll := m.coll.CopySession()
	defer coll.CloseSession()

	var (
		xl     = xlog.FromContextSafe(ctx)
		labels = []*LabelInMgo{}
	)

	err := coll.Find(bson.M{}).All(&labels)
	if err != nil {
		xl.Warnf("find label failed. %v", err)
	}

	return labels, err
}

func (m _LabelDAO) Insert(ctx context.Context, labels ...LabelInMgo) error {
	coll := m.coll.CopySession()
	defer coll.CloseSession()

	for _, label := range labels {

		label.ID = bson.NewObjectId()
		label.CreatedAt = time.Now()
		label.UpdatedAt = label.CreatedAt
		label.Version = 1

		err := coll.Insert(label)
		if err != nil {
			return err
		}
	}

	return nil
}

func (m _LabelDAO) Update(ctx context.Context, label *LabelInMgo) error {
	coll := m.coll.CopySession()
	defer coll.CloseSession()

	label.UpdatedAt = time.Now()
	version := label.Version
	label.Version++

	return coll.Update(bson.M{"_id": label.ID, "version": version}, label)
}

func (m _LabelDAO) Remove(ctx context.Context, name string) error {
	coll := m.coll.CopySession()
	defer coll.CloseSession()

	return coll.Remove(bson.M{"name": name})
}
