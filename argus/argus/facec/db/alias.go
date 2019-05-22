package db

import (
	"time"

	"gopkg.in/mgo.v2"
	"gopkg.in/mgo.v2/bson"

	"github.com/qiniu/db/mgoutil.v3"
	xlog "github.com/qiniu/xlog.v1"
)

// Alias maping the collection alias
type Alias struct {
	ID      bson.ObjectId `json:"_id"`
	AliasID int64         `json:"alias_id"`
	Euid    string        `json:"euid"`
	GroupID int64         `json:"group_id"`
	UID     string        `json:"uid"`
}

//const aliasCollection = "alias"

// AliasDao alias collection dao
type AliasDao interface {
	FindGroup(uid, euid string, finalID int64) ([]Alias, error)
	FindAlias(uid, euid string, groupIDs ...int64) ([]int64, error)
	UpdateAlias(uid, euid string, destGID int64, candiateGIDs ...int64) error
}

// NewAliasDao new face dao
func NewAliasDao() (AliasDao, error) {
	return &aliasDao{coll: &collections.Alias}, nil
}

type aliasDao struct{ coll *mgoutil.Collection }

func (d *aliasDao) EnsureIndexes() error {
	coll := d.coll.CopySession()
	defer coll.CloseSession()

	if err := coll.EnsureIndex(mgo.Index{Key: []string{"uid", "euid", "group_id"}, Unique: true}); err != nil {
		return err
	}
	if err := coll.EnsureIndex(mgo.Index{Key: []string{"uid", "euid", "alias_id"}}); err != nil {
		return err
	}
	return nil
}

func (d *aliasDao) FindGroup(uid, euid string, finalID int64) ([]Alias, error) {
	c := d.coll.CopySession()
	defer c.CloseSession()

	var group []Alias
	err := c.Find(bson.M{
		"uid":      uid,
		"euid":     euid,
		"alias_id": finalID,
	}).All(&group)

	if err != nil {
		xlog.Errorf("", "find group error:%v", err)
		return nil, err
	}

	return group, nil
}

func (d *aliasDao) FindAlias(uid, euid string, groupIDs ...int64) ([]int64, error) {
	c := d.coll.CopySession()
	defer c.CloseSession()

	var alias []Alias
	cond := bson.M{
		"uid":  uid,
		"euid": euid,
	}

	if len(groupIDs) > 0 {
		cond["group_id"] = bson.M{
			"$in": groupIDs,
		}
	}

	err := c.Find(cond).Select(bson.M{"group_id": 1, "alias_id": 1}).All(&alias)
	if err != nil {
		xlog.Errorf("", "find alias error:%v", err)
		return nil, err
	}

	maxGroupID := int64(-1)
	for _, a := range alias {
		if a.GroupID > maxGroupID {
			maxGroupID = a.GroupID
		}
	}

	aliasIDs := make([]int64, maxGroupID+1)
	for i := int64(0); i <= maxGroupID; i++ {
		aliasIDs[i] = -1
	}

	for _, a := range alias {
		aliasIDs[a.GroupID] = a.AliasID
	}

	return aliasIDs, nil
}

func (d *aliasDao) UpdateAlias(uid, euid string, destGID int64, candiateGIDs ...int64) error {
	c := d.coll.CopySession()
	defer c.CloseSession()

	var bkalias []interface{}
	for _, cad := range candiateGIDs {
		bkalias = append(bkalias,
			bson.M{
				"uid":      uid,
				"euid":     euid,
				"group_id": cad,
			},
			bson.M{"$set": bson.M{
				"alias_id":   destGID,
				"created_at": time.Now()},
			})
	}
	bk := c.Bulk()
	bk.Upsert(bkalias...)
	_, err := bk.Run()
	if err != nil {
		xlog.Errorf("", "upsert items to alias error:%v", err)
		return err
	}

	return nil
}
