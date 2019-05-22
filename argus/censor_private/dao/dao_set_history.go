package dao

import (
	mgoutil "github.com/qiniu/db/mgoutil.v3"
	"gopkg.in/mgo.v2/bson"
	"qiniu.com/argus/censor_private/proto"
)

type ISetHistoryDAO interface {
	Insert(history *proto.SetHistory) (id string, err error)
	Query(setId string) ([]*proto.SetHistory, error)
	Remove(id string) error
	RemoveBySet(setId string) error
}

type SetHistoryDAOInMgo struct {
	c *mgoutil.Collection
}

func NewSetHistoryDAOInMgo(c *mgoutil.Collection) ISetHistoryDAO {
	return &SetHistoryDAOInMgo{c: c}
}

func (dao *SetHistoryDAOInMgo) Insert(history *proto.SetHistory) (id string, err error) {
	history.Id = bson.NewObjectId()

	query(dao.c, func(c *mgoutil.Collection) {
		err = c.Insert(history)
	})

	id = history.Id.Hex()
	return
}

func (dao *SetHistoryDAOInMgo) Query(setId string) (items []*proto.SetHistory, err error) {
	items = make([]*proto.SetHistory, 0)

	query(dao.c, func(c *mgoutil.Collection) {
		err = c.Find(bson.M{"set_id": setId}).Sort("-_id").All(&items)
	})

	return
}

func (dao *SetHistoryDAOInMgo) Remove(id string) (err error) {
	if !bson.IsObjectIdHex(id) {
		return proto.ErrSetHistoryNotExist
	}

	query(dao.c, func(c *mgoutil.Collection) {
		err = c.RemoveId(bson.ObjectIdHex(id))
	})
	return
}

func (dao *SetHistoryDAOInMgo) RemoveBySet(setId string) (err error) {
	query(dao.c, func(c *mgoutil.Collection) {
		_, err = c.RemoveAll(bson.M{"set_id": setId})
	})
	return
}
