package dao

import (
	"time"

	mgoutil "github.com/qiniu/db/mgoutil.v3"
	"github.com/qiniu/uuid"
	mgo "gopkg.in/mgo.v2"
	"gopkg.in/mgo.v2/bson"
	"qiniu.com/argus/censor_private/proto"
)

type ISetDAO interface {
	Find(id string) (*proto.Set, error)
	Insert(set *proto.Set) (id string, err error)
	Query(*SetFilter) ([]*proto.Set, error)
	Patch(id string, update bson.M) error
	Update(id string, set *proto.Set) error
	Count(*SetFilter) (int, error)
	Remove(id string) error
}

type SetDAOInMgo struct {
	c *mgoutil.Collection
}

func NewSetDAOInMgo(c *mgoutil.Collection) ISetDAO {
	return &SetDAOInMgo{c: c}
}

type SetFilter struct {
	Id       string          `json:"id"`
	Name     string          `json:"name"`
	MimeType proto.MimeType  `json:"mime_type"`
	Scene    proto.Scene     `json:"scene"`
	Uri      string          `json:"uri"`
	Status   proto.SetStatus `json:"status"`
	Type     proto.SetType   `json:"type"`
}

func (f *SetFilter) toQueryParams() bson.M {
	q := bson.M{}

	if f == nil {
		return q
	}

	if len(f.Id) > 0 {
		q["id"] = f.Id
	}
	if len(f.Name) > 0 {
		q["name"] = f.Name
	}
	if f.MimeType.IsValid() {
		q["mime_types"] = f.MimeType
	}
	if len(f.Uri) > 0 {
		q["uri"] = f.Uri
	}
	if f.Scene.IsValid() {
		q["scenes"] = f.Scene
	}
	if f.Status.IsValid() {
		q["status"] = f.Status
	}
	if f.Type.IsValid() {
		q["type"] = f.Type
	}
	return q
}

func (dao *SetDAOInMgo) Find(id string) (item *proto.Set, err error) {
	query(dao.c, func(c *mgoutil.Collection) {
		err = c.Find(bson.M{"id": id}).One(&item)
		if err == mgo.ErrNotFound {
			err = proto.ErrSetNotExist
		}
	})

	return
}

func (dao *SetDAOInMgo) Insert(set *proto.Set) (id string, err error) {
	if len(set.Id) == 0 {
		set.Id, err = uuid.Gen(16)
		if err != nil {
			return "", proto.ErrGenId
		}
	}
	set.CreatedAt = time.Now().Unix()

	query(dao.c, func(c *mgoutil.Collection) {
		err = c.Insert(set)
	})

	id = set.Id
	return
}

func (dao *SetDAOInMgo) Query(filter *SetFilter) (items []*proto.Set, err error) {
	items = make([]*proto.Set, 0)

	query(dao.c, func(c *mgoutil.Collection) {
		q := filter.toQueryParams()
		err = c.Find(q).Sort("-_id").All(&items)
	})

	return
}

func (dao *SetDAOInMgo) Remove(id string) (err error) {
	query(dao.c, func(c *mgoutil.Collection) {
		_, err = c.RemoveAll(bson.M{"id": id})
		if err == mgo.ErrNotFound {
			err = proto.ErrSetNotExist
		}
	})

	return
}

func (dao *SetDAOInMgo) Patch(id string, m bson.M) (err error) {
	m["modified_at"] = time.Now().Unix()

	query(dao.c, func(c *mgoutil.Collection) {
		err = c.Update(
			bson.M{"id": id},
			bson.M{"$set": m})
		if err == mgo.ErrNotFound {
			err = proto.ErrSetNotExist
		}
	})

	return
}

func (dao *SetDAOInMgo) Update(id string, set *proto.Set) (err error) {
	query(dao.c, func(c *mgoutil.Collection) {
		err = c.Update(bson.M{"id": id}, set)
		if err == mgo.ErrNotFound {
			err = proto.ErrEntryNotExist
		}
	})

	return
}

func (dao *SetDAOInMgo) Count(filter *SetFilter) (n int, err error) {
	query(dao.c, func(c *mgoutil.Collection) {
		q := filter.toQueryParams()
		n, err = c.Find(q).Count()
	})
	return
}
