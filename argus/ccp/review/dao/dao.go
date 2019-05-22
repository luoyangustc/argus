package dao

import (
	mgoutil "github.com/qiniu/db/mgoutil.v3"
	mgo "gopkg.in/mgo.v2"
	"gopkg.in/mgo.v2/bson"
)

var (
	colls struct {
		QiniuIncrementEntries mgoutil.Collection `coll:"qiniu_increment_entries"`
		QiniuInventoryEntries mgoutil.Collection `coll:"qiniu_inventory_entries"`
		ApiIncrementEntries   mgoutil.Collection `coll:"api_increment_entries"`
		ApiInventoryEntries   mgoutil.Collection `coll:"api_inventory_entries"`
		EntrySets             mgoutil.Collection `coll:"entry_sets"`
		SetCounters           mgoutil.Collection `coll:"set_counters"`
		BatchEntryJobs        mgoutil.Collection `coll:"batch_entry_jobs"`
		VideoCuts             mgoutil.Collection `coll:"video_cuts"`
	}

	QnIncEntriesDao, QnInvEntriesDao, ApiIncEntriesDao, ApiInvEntriesDao EntryDAO
	SetDao                                                               _SetDAO
	SetCounterDAO                                                        _SetCounterDAO
	BatchEntryJobDAO                                                     _BatchEntryJobDAO
	VideoCutDAO                                                          _VideoCutDAO
)

func SetUp(cfg *mgoutil.Config) (sess *mgo.Session, err error) {
	sess, err = mgoutil.Open(&colls, cfg)
	if err != nil {
		return
	}

	// maybe we should make it configuration
	sess.SetPoolLimit(100)

	var (
		defaultEntryIndex = mgo.Index{Key: []string{"set_id", "mimetype", "_id"}}
	)
	// init all colls index
	_ = colls.QiniuIncrementEntries.EnsureIndex(defaultEntryIndex)
	_ = colls.QiniuInventoryEntries.EnsureIndex(defaultEntryIndex)
	_ = colls.ApiIncrementEntries.EnsureIndex(defaultEntryIndex)
	_ = colls.ApiInventoryEntries.EnsureIndex(defaultEntryIndex)

	_ = colls.EntrySets.EnsureIndex(mgo.Index{Key: []string{"set_id"}, Unique: true})
	_ = colls.EntrySets.EnsureIndex(mgo.Index{Key: []string{"uid", "source_type", "type", "bucket", "prefix"}})

	_ = colls.SetCounters.EnsureIndex(mgo.Index{Key: []string{"set_id"}, Unique: true})
	_ = colls.SetCounters.EnsureIndex(mgo.Index{Key: []string{"uid", "resource_id"}})

	_ = colls.BatchEntryJobs.EnsureIndex(mgo.Index{Key: []string{"status"}})

	_ = colls.VideoCuts.EnsureIndex(mgo.Index{Key: []string{"entry_id"}})

	// init all dao
	QnIncEntriesDao = NewEntryDAOInMgo(&colls.QiniuIncrementEntries)
	QnInvEntriesDao = NewEntryDAOInMgo(&colls.QiniuInventoryEntries)
	ApiIncEntriesDao = NewEntryDAOInMgo(&colls.ApiIncrementEntries)
	ApiInvEntriesDao = NewEntryDAOInMgo(&colls.ApiInventoryEntries)
	SetDao = NewSetDAOInMgo(&colls.EntrySets)
	SetCounterDAO = NewSetCounterDAO(&colls.SetCounters)
	BatchEntryJobDAO = NewBatchEntryJobDAO(&colls.BatchEntryJobs)
	VideoCutDAO = NewVideoCutDAOInMgo(&colls.VideoCuts)

	return
}

func query(coll *mgoutil.Collection, q func(*mgoutil.Collection)) {
	copiedCollection := coll.CopySession()
	defer copiedCollection.CloseSession()

	q(&copiedCollection)
}

func getQueryParamsWithPaginator(q bson.M, p *Paginator, isgt ...bool) (bson.M, int) {
	limit := defaultPaginatorLimit

	if p != nil {
		if p.IsValid() {
			if len(isgt) > 0 && isgt[0] {
				q["_id"] = bson.M{
					"$gt": bson.ObjectIdHex(p.Marker),
				}
			} else {
				q["_id"] = bson.M{
					"$lt": bson.ObjectIdHex(p.Marker),
				}
			}
		}

		limit = p.Limit
	}

	return q, limit
}
