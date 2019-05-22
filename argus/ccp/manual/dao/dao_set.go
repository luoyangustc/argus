package dao

import (
	"context"
	"time"

	mgoutil "github.com/qiniu/db/mgoutil.v3"
	xlog "github.com/qiniu/xlog.v1"
	mgo "gopkg.in/mgo.v2"
	"gopkg.in/mgo.v2/bson"
)

type ISetDAO interface {
	QueryAll(ctx context.Context) ([]SetInMgo, error)
	QueryByID(ctx context.Context, setID string) (*SetInMgo, error)
	Insert(ctx context.Context, set *SetInMgo) error
	Update(ctx context.Context, set *SetInMgo) error
	Remove(ctx context.Context, setID string) error
}

var _ ISetDAO = _SetDAO{}

type _SetDAO struct {
	CcpCapMgoConfig
	coll mgoutil.Collection
}

func NewSetInMgo(conf CcpCapMgoConfig) (ISetDAO, error) {
	var (
		mgoSessionPoolLimit = 100
		colls               = struct {
			Set mgoutil.Collection `coll:"cap_set"`
		}{}
	)
	sess, err := mgoutil.Open(&colls, &conf.Mgo)
	if err != nil {
		return _SetDAO{}, err
	}
	if conf.MgoPoolLimit > 0 {
		mgoSessionPoolLimit = conf.MgoPoolLimit
	}
	sess.SetPoolLimit(mgoSessionPoolLimit)

	err = colls.Set.EnsureIndex(mgo.Index{Key: []string{"set_id"}, Unique: true})
	if err != nil {
		return nil, err
	}

	err = colls.Set.EnsureIndex(mgo.Index{Key: []string{"source_type"}})
	if err != nil {
		return nil, err
	}

	err = colls.Set.EnsureIndex(mgo.Index{Key: []string{"type"}})
	if err != nil {
		return nil, err
	}

	return _SetDAO{CcpCapMgoConfig: conf, coll: colls.Set}, nil
}

func (m _SetDAO) QueryAll(ctx context.Context) (resp []SetInMgo, err error) {
	coll := m.coll.CopySession()
	defer coll.CloseSession()

	var (
		xl = xlog.FromContextSafe(ctx)
	)
	xl.Infof("begin list all sets")

	err = coll.Find(bson.M{}).All(&resp)
	if err != nil {
		xl.Warnf("list all sets error: %v", err.Error())
		return nil, err
	}

	return resp, nil
}

func (m _SetDAO) QueryByID(ctx context.Context, setID string) (resp *SetInMgo, err error) {
	coll := m.coll.CopySession()
	defer coll.CloseSession()

	var (
		xl = xlog.FromContextSafe(ctx)
	)
	xl.Infof("begin find setId: %#v", setID)

	err = coll.Find(bson.M{"set_id": setID}).One(&resp)
	if err != nil {
		xl.Warnf("find set failed. %s %v", setID, err)
		return nil, err
	}

	return resp, nil
}

func (m _SetDAO) Insert(ctx context.Context, set *SetInMgo) error {
	coll := m.coll.CopySession()
	defer coll.CloseSession()
	var (
		xl = xlog.FromContextSafe(ctx)
	)
	xl.Infof("begin insert: %s", set.SetID)

	set.ID = bson.NewObjectId()
	set.CreatedAt = time.Now()
	set.UpdatedAt = set.CreatedAt

	err := coll.Insert(set)
	if err != nil {
		xl.Errorf("insert to DB error: %#v", err.Error())
		return err
	}

	return nil
}

func (m _SetDAO) Update(ctx context.Context, set *SetInMgo) error {
	coll := m.coll.CopySession()
	defer coll.CloseSession()
	var (
		xl = xlog.FromContextSafe(ctx)
	)
	xl.Infof("begin update set: %#v", set.SetID)

	set.UpdatedAt = time.Now()
	err := coll.Update(bson.M{"set_id": set.SetID}, set)
	if err != nil {
		xl.Warnf("update %s error: %#v", set.SetID, err.Error())
		return err
	}
	return nil
}

func (m _SetDAO) Remove(ctx context.Context, setID string) error {
	coll := m.coll.CopySession()
	defer coll.CloseSession()

	var (
		xl = xlog.FromContextSafe(ctx)
	)
	xl.Infof("begin remove set: %#v", setID)

	return coll.Remove(bson.M{"set_id": setID})
}
