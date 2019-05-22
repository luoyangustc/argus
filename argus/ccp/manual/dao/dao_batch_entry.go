package dao

import (
	"context"
	"time"

	mgoutil "github.com/qiniu/db/mgoutil.v3"
	xlog "github.com/qiniu/xlog.v1"
	mgo "gopkg.in/mgo.v2"
	"gopkg.in/mgo.v2/bson"
	"qiniu.com/argus/ccp/manual/enums"
)

type IBatchEntryDAO interface {
	QueryByID(context.Context, string) (*BatchEntryInMgo, error)
	QueryByImageSetID(context.Context, string) (*BatchEntryInMgo, error)
	QueryByVideoSetID(context.Context, string) (*BatchEntryInMgo, error)
	QueryByStatus(context.Context, string) ([]*BatchEntryInMgo, error)
	BatchInsert(context.Context, *BatchEntryInMgo) error
	StartJob(context.Context, string) error
	UpdateStatus(context.Context, string, string) error
	Remove(context.Context, string) error
}

var _ IBatchEntryDAO = _BatchEntryDAO{}

type _BatchEntryDAO struct {
	CcpCapMgoConfig
	coll mgoutil.Collection
}

func NewBatchEntryInMgo(conf CcpCapMgoConfig) (IBatchEntryDAO, error) {
	var (
		mgoSessionPoolLimit = 100
		colls               = struct {
			Entry mgoutil.Collection `coll:"cap_batch_entry"`
		}{}
	)
	sess, err := mgoutil.Open(&colls, &conf.Mgo)
	if err != nil {
		return _BatchEntryDAO{}, err
	}
	if conf.MgoPoolLimit > 0 {
		mgoSessionPoolLimit = conf.MgoPoolLimit
	}
	sess.SetPoolLimit(mgoSessionPoolLimit)

	err = colls.Entry.EnsureIndex(mgo.Index{Key: []string{"set_id"}})
	if err != nil {
		return nil, err
	}

	err = colls.Entry.EnsureIndex(mgo.Index{Key: []string{"image_set_id"}})
	if err != nil {
		return nil, err
	}

	err = colls.Entry.EnsureIndex(mgo.Index{Key: []string{"video_set_id"}})
	if err != nil {
		return nil, err
	}

	err = colls.Entry.EnsureIndex(mgo.Index{Key: []string{"status"}})
	if err != nil {
		return nil, err
	}

	return _BatchEntryDAO{CcpCapMgoConfig: conf, coll: colls.Entry}, nil
}

func (this _BatchEntryDAO) QueryByID(ctx context.Context, id string) (item *BatchEntryInMgo, err error) {
	coll := this.coll.CopySession()
	defer coll.CloseSession()

	var (
		xl = xlog.FromContextSafe(ctx)
	)
	xl.Infof("begin query entry by id: %s", id)

	err = coll.Find(bson.M{
		"set_id": id,
	}).One(&item)

	return
}

func (this _BatchEntryDAO) QueryByImageSetID(ctx context.Context, id string) (item *BatchEntryInMgo, err error) {
	coll := this.coll.CopySession()
	defer coll.CloseSession()

	var (
		xl = xlog.FromContextSafe(ctx)
	)
	xl.Infof("begin query entry by imageId: %s", id)

	err = coll.Find(bson.M{
		"image_set_id": id,
	}).One(&item)

	return
}
func (this _BatchEntryDAO) QueryByVideoSetID(ctx context.Context, id string) (item *BatchEntryInMgo, err error) {
	coll := this.coll.CopySession()
	defer coll.CloseSession()

	var (
		xl = xlog.FromContextSafe(ctx)
	)
	xl.Infof("begin query entry by videoId: %s", id)

	err = coll.Find(bson.M{
		"video_set_id": id,
	}).One(&item)

	return
}

func (this _BatchEntryDAO) QueryByStatus(ctx context.Context,
	status string) (items []*BatchEntryInMgo, err error) {
	coll := this.coll.CopySession()
	defer coll.CloseSession()

	err = coll.Find(bson.M{
		"status": status,
	}).Limit(20).All(&items)

	return

}

func (this _BatchEntryDAO) BatchInsert(ctx context.Context, req *BatchEntryInMgo) (err error) {
	coll := this.coll.CopySession()
	defer coll.CloseSession()

	var (
		xl = xlog.FromContextSafe(ctx)
	)
	xl.Infof("begin insert batch entryJobDao: %#v", req.SetId)

	req.ID = bson.NewObjectId()
	req.CreatedAt = time.Now()
	req.UpdatedAt = req.CreatedAt

	err = coll.Insert(req)
	if err != nil {
		return err
	}

	return
}

func (this _BatchEntryDAO) StartJob(ctx context.Context, id string) (err error) {
	coll := this.coll.CopySession()
	defer coll.CloseSession()

	var (
		xl = xlog.FromContextSafe(ctx)
	)
	xl.Infof("begin StartJob: %s", id)
	err = coll.Update(bson.M{
		"set_id": id,
		"status": enums.BatchEntryJobStatusNew,
	}, bson.M{
		"$set": bson.M{
			"status":     enums.BatchEntryJobStatusProcess,
			"updated_at": time.Now(),
			"start_at":   time.Now().Unix(),
		},
	})
	if err != nil {
		xl.Errorf("startJob error: %#v", err.Error())
		return err
	}
	return nil
}

func (this _BatchEntryDAO) UpdateStatus(ctx context.Context, setId string, status string) (err error) {
	coll := this.coll.CopySession()
	defer coll.CloseSession()

	var (
		xl = xlog.FromContextSafe(ctx)
	)
	xl.Infof("update batchJob status: %s to %s", setId, status)
	err = coll.Update(bson.M{
		"set_id": setId,
	}, bson.M{
		"$set": bson.M{
			"status": status,
		},
	})
	if err != nil {
		xl.Errorf("UpdateStatus error: %#v", err.Error())
		return err
	}

	return nil
}

func (this _BatchEntryDAO) Remove(ctx context.Context, setID string) error {
	coll := this.coll.CopySession()
	defer coll.CloseSession()

	var (
		xl = xlog.FromContextSafe(ctx)
	)
	xl.Infof("begin remove batchEntry: %#v", setID)

	return coll.Remove(bson.M{"set_id": setID})
}
