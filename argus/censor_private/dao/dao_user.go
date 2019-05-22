package dao

import (
	"time"

	mgoutil "github.com/qiniu/db/mgoutil.v3"
	mgo "gopkg.in/mgo.v2"
	"gopkg.in/mgo.v2/bson"
	"qiniu.com/argus/censor_private/proto"
)

type IUserDAO interface {
	Find(id string) (*proto.User, error)
	Insert(user *proto.User) (err error)
	Query(keyword string) ([]*proto.User, error)
	Patch(id string, update bson.M) error
	Update(id string, user *proto.User) error
	Remove(id string) error
}

type UserDAOInMgo struct {
	c *mgoutil.Collection
}

func NewUserDAOInMgo(c *mgoutil.Collection) IUserDAO {
	return &UserDAOInMgo{c: c}
}

func (dao *UserDAOInMgo) Find(id string) (item *proto.User, err error) {
	query(dao.c, func(c *mgoutil.Collection) {
		err = c.Find(bson.M{"id": id}).One(&item)
		if err == mgo.ErrNotFound {
			err = proto.ErrUserNotExist
		}
	})

	return
}

func (dao *UserDAOInMgo) Insert(user *proto.User) (err error) {
	user.CreatedAt = time.Now().Unix()

	query(dao.c, func(c *mgoutil.Collection) {
		err = c.Insert(user)
	})

	return
}

func (dao *UserDAOInMgo) Query(keyword string) (items []*proto.User, err error) {
	items = make([]*proto.User, 0)

	query(dao.c, func(c *mgoutil.Collection) {
		q := bson.M{}
		if len(keyword) > 0 {
			regex := bson.RegEx{
				Pattern: keyword,
				Options: "i",
			}
			q["$or"] = []bson.M{bson.M{"id": regex}, bson.M{"desc": regex}}
		}
		err = c.Find(q).Sort("-_id").All(&items)
	})

	return
}

func (dao *UserDAOInMgo) Remove(id string) (err error) {
	query(dao.c, func(c *mgoutil.Collection) {
		_, err = c.RemoveAll(bson.M{"id": id})
		if err == mgo.ErrNotFound {
			err = proto.ErrUserNotExist
		}
	})

	return
}

func (dao *UserDAOInMgo) Patch(id string, m bson.M) (err error) {
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

func (dao *UserDAOInMgo) Update(id string, user *proto.User) (err error) {
	query(dao.c, func(c *mgoutil.Collection) {
		err = c.Update(bson.M{"id": id}, user)
		if err == mgo.ErrNotFound {
			err = proto.ErrEntryNotExist
		}
	})

	return
}
