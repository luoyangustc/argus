package dao

import (
	mgoutil "github.com/qiniu/db/mgoutil.v3"
	mgo "gopkg.in/mgo.v2"
)

var (
	colls struct {
		Entries     mgoutil.Collection `coll:"entries"`
		Sets        mgoutil.Collection `coll:"sets"`
		SetsHistory mgoutil.Collection `coll:"sets_history"`
		Users       mgoutil.Collection `coll:"users"`
		VideoCuts   mgoutil.Collection `coll:"video_cuts"`
	}

	EntryDao      IEntryDAO
	SetDao        ISetDAO
	SetHistoryDao ISetHistoryDAO
	UserDao       IUserDAO
	VideoCutDao   IVideoCutDAO
)

func SetUp(cfg *mgoutil.Config) (sess *mgo.Session, err error) {
	sess, err = mgoutil.Open(&colls, cfg)
	if err != nil {
		return
	}

	sess.SetPoolLimit(100)

	// init index
	_ = colls.Entries.EnsureIndex(mgo.Index{Key: []string{"set_id", "_id", "created_at"}, Background: true})
	_ = colls.Sets.EnsureIndex(mgo.Index{Key: []string{"id"}, Unique: true})
	_ = colls.SetsHistory.EnsureIndex(mgo.Index{Key: []string{"set_id", "_id"}})
	_ = colls.Users.EnsureIndex(mgo.Index{Key: []string{"id"}, Unique: true})
	_ = colls.VideoCuts.EnsureIndex(mgo.Index{Key: []string{"entry_id"}})

	// init dao
	EntryDao = NewEntryDAOInMgo(&colls.Entries)
	SetDao = NewSetDAOInMgo(&colls.Sets)
	SetHistoryDao = NewSetHistoryDAOInMgo(&colls.SetsHistory)
	UserDao = NewUserDAOInMgo(&colls.Users)
	VideoCutDao = NewVideoCutDAOInMgo(&colls.VideoCuts)

	return
}

func query(coll *mgoutil.Collection, q func(*mgoutil.Collection)) {
	copiedCollection := coll.CopySession()
	defer copiedCollection.CloseSession()

	q(&copiedCollection)
}
