package dao

import (
	"context"
	"time"

	mgoutil "github.com/qiniu/db/mgoutil.v3"
	xlog "github.com/qiniu/xlog.v1"
	mgo "gopkg.in/mgo.v2"
	"gopkg.in/mgo.v2/bson"
)

type IEntryDAO interface {
	QueryBySetId(ctx context.Context, setID string, p *Paginator) ([]*EntryInMgo, error)
	QueryByID(ctx context.Context, setId, entryId string) (*EntryInMgo, error)
	Insert(ctx context.Context, entry *EntryInMgo) error
	Update(ctx context.Context, entry *EntryInMgo) error
	Remove(ctx context.Context, id string) error
}

var _ IEntryDAO = _EntyrDAO{}

type _EntyrDAO struct {
	CcpCapMgoConfig
	coll mgoutil.Collection
}

func NewEntryInMgo(conf CcpCapMgoConfig) (IEntryDAO, error) {
	var (
		mgoSessionPoolLimit = 100
		colls               = struct {
			Entry mgoutil.Collection `coll:"cap_stream_entry"`
		}{}
	)
	sess, err := mgoutil.Open(&colls, &conf.Mgo)
	if err != nil {
		return _EntyrDAO{}, err
	}
	if conf.MgoPoolLimit > 0 {
		mgoSessionPoolLimit = conf.MgoPoolLimit
	}
	sess.SetPoolLimit(mgoSessionPoolLimit)

	err = colls.Entry.EnsureIndex(mgo.Index{Key: []string{"set_id"}})
	if err != nil {
		return nil, err
	}

	err = colls.Entry.EnsureIndex(mgo.Index{Key: []string{"entry_id"}})
	if err != nil {
		return nil, err
	}

	err = colls.Entry.EnsureIndex(mgo.Index{Key: []string{"created_at"}})
	if err != nil {
		return nil, err
	}

	err = colls.Entry.EnsureIndex(mgo.Index{Key: []string{"updated_at"}})
	if err != nil {
		return nil, err
	}

	return _EntyrDAO{CcpCapMgoConfig: conf, coll: colls.Entry}, nil
}

func (m _EntyrDAO) QueryBySetId(ctx context.Context, setID string, p *Paginator) (resp []*EntryInMgo, err error) {
	coll := m.coll.CopySession()
	defer coll.CloseSession()

	var (
		xl    = xlog.FromContextSafe(ctx)
		q     = bson.M{"set_id": setID}
		limit = defaultPaginatorLimit
		skip  = 0
	)

	if p != nil {
		if p.IsValid() {
			skip = p.Offset
			limit = p.Limit
		}
	}

	xl.Infof("begin ListBySetId")
	err = coll.Find(q).Sort("-_id").Skip(skip).Limit(limit).All(&resp)
	if err != nil {
		xl.Warnf("find set failed. %s %v", setID, err)
	}

	return resp, err
}

func (m _EntyrDAO) QueryByID(ctx context.Context, setId, entryId string) (resp *EntryInMgo, err error) {
	coll := m.coll.CopySession()
	defer coll.CloseSession()

	var (
		xl = xlog.FromContextSafe(ctx)
	)
	xl.Infof("begin query entry by id: %s", setId)

	err = coll.Find(bson.M{"set_id": setId, "entry_id": entryId}).One(&resp)
	if err != nil {
		xl.Warnf("query entry failed. %s %v", entryId, err)
	}

	return resp, err
}

func (m _EntyrDAO) Insert(ctx context.Context, entry *EntryInMgo) error {
	coll := m.coll.CopySession()
	defer coll.CloseSession()
	var (
		xl = xlog.FromContextSafe(ctx)
	)
	xl.Infof("begin insert")
	entry.ID = bson.NewObjectId()
	entry.CreatedAt = time.Now()
	entry.UpdatedAt = entry.CreatedAt

	err := coll.Insert(entry)
	if err != nil {
		return err
	}

	return nil
}

func (m _EntyrDAO) Update(ctx context.Context, entry *EntryInMgo) error {
	coll := m.coll.CopySession()
	defer coll.CloseSession()
	var (
		xl = xlog.FromContextSafe(ctx)
	)
	xl.Infof("begin update")

	entry.UpdatedAt = time.Now()
	err := coll.Update(bson.M{"_id": entry.ID}, entry)
	if err != nil {
		xl.Warnf("update %s error: %#v", entry.EntryID, err.Error())
		return err
	}
	return nil
}

func (m _EntyrDAO) Remove(ctx context.Context, id string) error {
	coll := m.coll.CopySession()
	defer coll.CloseSession()

	return coll.Remove(bson.M{"entry_id": id})
}
