package dao

import (
	"context"

	mgoutil "github.com/qiniu/db/mgoutil.v3"
	"gopkg.in/mgo.v2/bson"
	"qiniu.com/argus/ccp/review/model"
)

type _VideoCutDAO interface {
	BatchInsert(context.Context, []*model.VideoCut) error
	Query(context.Context, string, *Paginator) ([]*model.VideoCut, error)
	Count(context.Context, string) (int, error)
}

type VideoCutDAOInMgo struct {
	c *mgoutil.Collection
}

func NewVideoCutDAOInMgo(c *mgoutil.Collection) _VideoCutDAO {
	return &VideoCutDAOInMgo{c: c}
}

func (this *VideoCutDAOInMgo) Query(ctx context.Context, entryId string, p *Paginator) (items []*model.VideoCut, err error) {
	if entryId == "" || !bson.IsObjectIdHex(entryId) {
		return nil, ErrInvalidId
	}

	items = make([]*model.VideoCut, 0)

	query(this.c, func(c *mgoutil.Collection) {
		q, limit := getQueryParamsWithPaginator(bson.M{
			"entry_id": entryId,
		}, p, true)
		err = c.Find(q).Sort("_id").Limit(limit).All(&items)
	})

	return
}

func (this *VideoCutDAOInMgo) BatchInsert(ctx context.Context, cuts []*model.VideoCut) (err error) {
	if len(cuts) == 0 {
		return
	}

	docs := make([]interface{}, len(cuts))

	for i, cut := range cuts {
		cut.ID = bson.NewObjectId()
		docs[i] = cut
	}

	query(this.c, func(c *mgoutil.Collection) {
		err = c.Insert(docs...)
	})
	return
}

func (this *VideoCutDAOInMgo) Count(ctx context.Context, entryId string) (count int, err error) {
	if !bson.IsObjectIdHex(entryId) {
		return 0, ErrInvalidId
	}

	query(this.c, func(c *mgoutil.Collection) {
		count, err = c.Find(bson.M{
			"entry_id": entryId,
		}).Count()
	})

	return
}
