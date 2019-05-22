package dao

import (
	"fmt"

	"qiniu.com/argus/censor_private/proto"

	mgoutil "github.com/qiniu/db/mgoutil.v3"
	mgo "gopkg.in/mgo.v2"
	"gopkg.in/mgo.v2/bson"
)

type IEntryDAO interface {
	Query(filter *EntryFilter, marker string, limit int) ([]*proto.Entry, string, error)
	FindPending(setId, markerStart, markerEnd string, limit int) ([]*proto.Entry, error)
	Find(id string) (*proto.Entry, error)
	Insert(*proto.Entry) (id string, err error)
	BatchInsert([]*proto.Entry) error
	Update(id string, entry *proto.Entry) error
	Patch(id string, update bson.M) error
	PatchMulti(ids []string, update bson.M) error
	Count(*EntryFilter) (int, error)
	Remove(id string) error
	RemoveBySet(setId string) error
}

type EntryDAOInMgo struct {
	c *mgoutil.Collection
}

func NewEntryDAOInMgo(c *mgoutil.Collection) IEntryDAO {
	return &EntryDAOInMgo{c: c}
}

type EntryFilter struct {
	SetId      string           `json:"set_id"`
	Mimetype   proto.MimeType   `json:"mime_type"`
	Suggestion proto.Suggestion `json:"suggestion"`
	Scenes     []proto.Scene    `json:"scenes"`
	StartAt    int64            `json:"start"`
	EndAt      int64            `json:"end"`
}

func (f *EntryFilter) toQueryParams() bson.M {
	q := bson.M{}

	if f == nil {
		return q
	}

	if len(f.SetId) != 0 {
		q["set_id"] = f.SetId
	}

	if f.StartAt != 0 && f.EndAt != 0 {
		q["created_at"] = bson.M{
			"$gte": f.StartAt,
			"$lte": f.EndAt,
		}
	}

	if f.Mimetype.IsValid() {
		q["mime_type"] = f.Mimetype
	}

	if f.Suggestion.IsValid() {
		// 取出满足以下条件之一的：
		// 1. 机审为该suggestion，且未人审的
		// 2. 人审为该suggestion的

		q11 := bson.M{"original.suggestion": f.Suggestion, "final": nil}
		qScenes := []bson.M{}
		for _, v := range f.Scenes {
			if v.IsValid() {
				ssKey := fmt.Sprintf("original.scenes.%s.suggestion", v)
				qScenes = append(qScenes, bson.M{ssKey: f.Suggestion})
			}
		}
		q1 := bson.M{}
		if len(qScenes) > 0 {
			q1["$and"] = []bson.M{q11, bson.M{"$or": qScenes}}
		} else {
			q1 = q11
		}

		q2 := bson.M{"final.suggestion": f.Suggestion}

		q["$or"] = []bson.M{q1, q2}
	} else if f.Suggestion == proto.SuggestionAll {
		// 选出所有机审过的，自然也包含人审过的
		q["original"] = bson.M{"$ne": nil}
	}

	return q
}

func (dao *EntryDAOInMgo) Query(filter *EntryFilter, marker string, limit int) (items []*proto.Entry, nextMarker string, err error) {
	items = make([]*proto.Entry, 0)

	query(dao.c, func(c *mgoutil.Collection) {
		q := filter.toQueryParams()
		if limit <= 0 {
			err = c.Find(q).Sort("-_id").All(&items)
		} else {
			if marker != "" {
				if !bson.IsObjectIdHex(marker) {
					err = proto.ErrInvalidMarker
					return
				}
				q["_id"] = bson.M{"$lt": bson.ObjectIdHex(marker)}
			}

			// 取出limit+1个是为了确认之后已经没有了, nextMarker须为空
			err = c.Find(q).Sort("-_id").Limit(limit + 1).All(&items)
			if len(items) > limit {
				nextMarker = items[limit-1].Id.Hex()
				items = items[:limit]
			}
		}
	})

	return
}

func (dao *EntryDAOInMgo) FindPending(setId, markerStart, markerEnd string, limit int) (items []*proto.Entry, err error) {
	items = make([]*proto.Entry, 0)

	q := bson.M{"original": nil, "error": nil}
	if len(setId) > 0 {
		q["set_id"] = setId
	}
	if bson.IsObjectIdHex(markerStart) {
		q["_id"] = bson.M{"$gt": bson.ObjectIdHex(markerStart)}
	}
	if bson.IsObjectIdHex(markerEnd) {
		q["_id"] = bson.M{"$lte": bson.ObjectIdHex(markerEnd)}
	}

	if limit <= 0 {
		limit = defaultPaginatorLimit
	}

	query(dao.c, func(c *mgoutil.Collection) {
		err = c.Find(q).Sort("_id").Limit(limit).All(&items)
	})

	return
}

func (dao *EntryDAOInMgo) Find(id string) (item *proto.Entry, err error) {
	if !bson.IsObjectIdHex(id) {
		return nil, proto.ErrEntryNotExist
	}

	query(dao.c, func(c *mgoutil.Collection) {
		err = c.FindId(bson.ObjectIdHex(id)).One(&item)
		if err == mgo.ErrNotFound {
			err = proto.ErrEntryNotExist
		}
	})

	return
}

func (dao *EntryDAOInMgo) Insert(entry *proto.Entry) (id string, err error) {
	entry.Id = bson.NewObjectId() // generate new ID

	query(dao.c, func(c *mgoutil.Collection) {
		err = c.Insert(entry)
	})

	id = entry.Id.Hex()
	return
}

func (dao *EntryDAOInMgo) BatchInsert(entries []*proto.Entry) (err error) {
	if len(entries) == 0 {
		return nil
	}

	docs := make([]interface{}, len(entries))
	for i, entry := range entries {
		entry.Id = bson.NewObjectId()
		docs[i] = entry
	}

	query(dao.c, func(c *mgoutil.Collection) {
		err = c.Insert(docs...)
	})
	return
}

func (dao *EntryDAOInMgo) Update(id string, entry *proto.Entry) (err error) {
	if !bson.IsObjectIdHex(id) {
		return proto.ErrEntryNotExist
	}

	query(dao.c, func(c *mgoutil.Collection) {
		err = c.Update(bson.M{"_id": bson.ObjectIdHex(id)}, entry)
		if err == mgo.ErrNotFound {
			err = proto.ErrEntryNotExist
		}
	})

	return
}

func (dao *EntryDAOInMgo) Patch(id string, m bson.M) (err error) {
	if !bson.IsObjectIdHex(id) {
		return proto.ErrEntryNotExist
	}

	query(dao.c, func(c *mgoutil.Collection) {
		err = c.Update(
			bson.M{"_id": bson.ObjectIdHex(id)},
			bson.M{"$set": m})
		if err == mgo.ErrNotFound {
			err = proto.ErrEntryNotExist
		}
	})

	return
}

func (dao *EntryDAOInMgo) PatchMulti(ids []string, m bson.M) (err error) {
	objectIds := make([]bson.ObjectId, 0)
	for _, v := range ids {
		if bson.IsObjectIdHex(v) {
			objectIds = append(objectIds, bson.ObjectIdHex(v))
		}
	}

	if len(objectIds) == 0 {
		return nil
	}

	query(dao.c, func(c *mgoutil.Collection) {
		_, err = c.UpdateAll(
			bson.M{"_id": bson.M{"$in": objectIds}},
			bson.M{"$set": m})
		if err == mgo.ErrNotFound {
			err = proto.ErrEntryNotExist
		}
	})

	return
}

func (dao *EntryDAOInMgo) Count(filter *EntryFilter) (n int, err error) {
	query(dao.c, func(c *mgoutil.Collection) {
		q := filter.toQueryParams()
		n, err = c.Find(q).Count()
	})
	return
}

func (dao *EntryDAOInMgo) Remove(id string) (err error) {
	if !bson.IsObjectIdHex(id) {
		return proto.ErrEntryNotExist
	}

	query(dao.c, func(c *mgoutil.Collection) {
		err = c.RemoveId(bson.ObjectIdHex(id))
	})
	return
}

func (dao *EntryDAOInMgo) RemoveBySet(setId string) (err error) {
	query(dao.c, func(c *mgoutil.Collection) {
		_, err = c.RemoveAll(bson.M{"set_id": setId})
	})
	return
}
