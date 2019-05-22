package dao

import (
	"context"

	mgoutil "github.com/qiniu/db/mgoutil.v3"
	"gopkg.in/mgo.v2/bson"
	"qiniu.com/argus/ccp/review/model"
)

type _SetCounterDAO interface {
	Query(ctx context.Context, uid uint32, setIds []string) ([]*model.SetCounter, error)
	QueryByResourceID(ctx context.Context, uid uint32, resoruceIds []string) ([]*model.SetCounter, error)
	Find(ctx context.Context, setId string) (*model.SetCounter, error)
	Insert(ctx context.Context, counter *model.SetCounter) error
	Update(ctx context.Context, counter *model.SetCounter, oldVersion int) error
	Remove(ctx context.Context, uid uint32, setId string) error
}

type SetCounterDAOInMgo struct {
	c *mgoutil.Collection
}

func NewSetCounterDAO(c *mgoutil.Collection) _SetCounterDAO {
	return &SetCounterDAOInMgo{c: c}
}

func (this *SetCounterDAOInMgo) Query(ctx context.Context, uid uint32, setIds []string) (items []*model.SetCounter, err error) {
	query(this.c, func(c *mgoutil.Collection) {
		q := bson.M{
			"uid": uid,
			"set_id": bson.M{
				"$in": setIds,
			},
		}

		err = c.Find(q).All(&items)
	})

	return
}

func (this *SetCounterDAOInMgo) QueryByResourceID(ctx context.Context, uid uint32, resourceIds []string) (items []*model.SetCounter, err error) {
	query(this.c, func(c *mgoutil.Collection) {
		q := bson.M{
			"uid": uid,
			"resource_id": bson.M{
				"$in": resourceIds,
			},
		}
		err = c.Find(q).All(&items)
	})
	return
}

func (this *SetCounterDAOInMgo) Find(ctx context.Context, setId string) (item *model.SetCounter, err error) {
	query(this.c, func(c *mgoutil.Collection) {
		err = c.Find(bson.M{"set_id": setId}).One(&item)
	})

	return
}

func (this *SetCounterDAOInMgo) Insert(ctx context.Context, counter *model.SetCounter) (err error) {
	counter.ID = bson.NewObjectId() // generate new ID
	query(this.c, func(c *mgoutil.Collection) {
		err = c.Insert(counter)
	})
	return
}

func (this *SetCounterDAOInMgo) Update(ctx context.Context, counter *model.SetCounter, oldVersion int) (err error) {
	query(this.c, func(c *mgoutil.Collection) {
		q := bson.M{
			"_id": counter.ID,
		}

		// FIX ME: for old data
		if oldVersion != 0 {
			q["version"] = oldVersion
		}

		err = c.Update(q, counter)
	})
	return
}

func (this *SetCounterDAOInMgo) Remove(ctx context.Context, uid uint32, setId string) (err error) {
	query(this.c, func(c *mgoutil.Collection) {
		err = c.Remove(bson.M{
			"uid":    uid,
			"set_id": setId,
		})
	})
	return
}
