package dao

import (
	"fmt"

	mgoutil "github.com/qiniu/db/mgoutil.v3"
	mgo "gopkg.in/mgo.v2"
	"gopkg.in/mgo.v2/bson"
	"qiniu.com/argus/censor_private/proto"
)

type IVideoCutDAO interface {
	Find(id string) (*proto.VideoCut, error)
	Insert(cut *proto.VideoCut) (err error)
	BatchInsert([]*proto.VideoCut) error
	Query(filter *VideoCutFilter, marker string, limit int) ([]*proto.VideoCut, string, error)
	Patch(id string, update bson.M) error
	Update(id string, cut *proto.VideoCut) error
	Remove(id string) error
	RemoveByEntry(id string) error
	Count(*VideoCutFilter) (int, error)
}

type VideoCutDAOInMgo struct {
	c *mgoutil.Collection
}

func NewVideoCutDAOInMgo(c *mgoutil.Collection) IVideoCutDAO {
	return &VideoCutDAOInMgo{c: c}
}

type VideoCutFilter struct {
	EntryId    string           `json:"entry_id"`
	Suggestion proto.Suggestion `json:"suggestion"`
	Scene      proto.Scene      `json:"scene"`
}

func (f *VideoCutFilter) toQueryParams() bson.M {
	q := bson.M{}

	if f == nil {
		return q
	}

	if len(f.EntryId) != 0 {
		q["entry_id"] = f.EntryId
	}

	if f.Suggestion.IsValid() {
		q["original.suggestion"] = f.Suggestion
		if f.Scene.IsValid() {
			ssKey := fmt.Sprintf("original.scenes.%s.suggestion", f.Scene)
			q[ssKey] = f.Suggestion
		}
	} else if f.Suggestion == proto.SuggestionAll {
		q["original"] = bson.M{"$ne": nil}
	}

	return q
}

func (dao *VideoCutDAOInMgo) Find(id string) (item *proto.VideoCut, err error) {
	if !bson.IsObjectIdHex(id) {
		return nil, proto.ErrVideoCutNotExist
	}

	query(dao.c, func(c *mgoutil.Collection) {
		err = c.FindId(bson.ObjectIdHex(id)).One(&item)
		if err == mgo.ErrNotFound {
			err = proto.ErrVideoCutNotExist
		}
	})

	return
}

func (dao *VideoCutDAOInMgo) Insert(cut *proto.VideoCut) (err error) {
	cut.Id = bson.NewObjectId() // generate new ID
	query(dao.c, func(c *mgoutil.Collection) {
		err = c.Insert(cut)
	})

	return
}

func (dao *VideoCutDAOInMgo) BatchInsert(cuts []*proto.VideoCut) (err error) {
	if len(cuts) == 0 {
		return nil
	}

	docs := make([]interface{}, len(cuts))
	for i, cut := range cuts {
		cut.Id = bson.NewObjectId()
		docs[i] = cut
	}

	query(dao.c, func(c *mgoutil.Collection) {
		err = c.Insert(docs...)
	})
	return
}

func (dao *VideoCutDAOInMgo) Query(filter *VideoCutFilter, marker string, limit int) (items []*proto.VideoCut, nextMarker string, err error) {
	items = make([]*proto.VideoCut, 0)

	query(dao.c, func(c *mgoutil.Collection) {
		q := filter.toQueryParams()
		if limit <= 0 {
			err = c.Find(q).Sort("_id").All(&items)
		} else {
			if marker != "" {
				if !bson.IsObjectIdHex(marker) {
					err = proto.ErrInvalidMarker
					return
				}
				q["_id"] = bson.M{"$gt": bson.ObjectIdHex(marker)}
			}

			// 取出limit+1个是为了确认之后已经没有了, nextMarker须为空
			err = c.Find(q).Sort("_id").Limit(limit + 1).All(&items)
			if len(items) > limit {
				nextMarker = items[limit-1].Id.Hex()
				items = items[:limit]
			}
		}
	})

	return
}

func (dao *VideoCutDAOInMgo) Update(id string, cut *proto.VideoCut) (err error) {
	if !bson.IsObjectIdHex(id) {
		return proto.ErrVideoCutNotExist
	}

	query(dao.c, func(c *mgoutil.Collection) {
		err = c.Update(bson.M{"_id": bson.ObjectIdHex(id)}, cut)
		if err == mgo.ErrNotFound {
			err = proto.ErrVideoCutNotExist
		}
	})

	return
}

func (dao *VideoCutDAOInMgo) Patch(id string, m bson.M) (err error) {
	if !bson.IsObjectIdHex(id) {
		return proto.ErrVideoCutNotExist
	}

	query(dao.c, func(c *mgoutil.Collection) {
		err = c.Update(
			bson.M{"_id": bson.ObjectIdHex(id)},
			bson.M{"$set": m})
		if err == mgo.ErrNotFound {
			err = proto.ErrVideoCutNotExist
		}
	})

	return
}
func (dao *VideoCutDAOInMgo) Remove(id string) (err error) {
	if !bson.IsObjectIdHex(id) {
		return proto.ErrVideoCutNotExist
	}

	query(dao.c, func(c *mgoutil.Collection) {
		err = c.RemoveId(bson.ObjectIdHex(id))
	})
	return
}

func (dao *VideoCutDAOInMgo) RemoveByEntry(entryId string) (err error) {
	query(dao.c, func(c *mgoutil.Collection) {
		_, err = c.RemoveAll(bson.M{"entry_id": entryId})
	})
	return
}

func (dao *VideoCutDAOInMgo) Count(filter *VideoCutFilter) (n int, err error) {
	query(dao.c, func(c *mgoutil.Collection) {
		q := filter.toQueryParams()
		n, err = c.Find(q).Count()
	})
	return
}
