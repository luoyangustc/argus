package dao

import (
	"context"
	"time"

	mgoutil "github.com/qiniu/db/mgoutil.v3"
	"gopkg.in/mgo.v2/bson"
	"qiniu.com/argus/ccp/review/enums"
	"qiniu.com/argus/ccp/review/model"
)

type _BatchEntryJobDAO interface {
	Find(context.Context, string) (*model.BatchEntryJob, error)
	Query(context.Context, enums.BatchEntryJobStatus) ([]*model.BatchEntryJob, error)
	BatchInsert(context.Context, []*model.BatchEntryJob) error
	StartJob(context.Context, bson.ObjectId) error
	UpdateLineNumber(context.Context, bson.ObjectId, int64) error
	UpdateStatus(context.Context, bson.ObjectId, enums.BatchEntryJobStatus, enums.BatchEntryJobStatus) error
	UpdateStatusBySetId(context.Context, string, enums.BatchEntryJobStatus) error
	RemoveBySetId(ctx context.Context, setId string) error
}

type BatchEntryJobDAOInMgo struct {
	c *mgoutil.Collection
}

func NewBatchEntryJobDAO(c *mgoutil.Collection) _BatchEntryJobDAO {
	return &BatchEntryJobDAOInMgo{c: c}
}

func (this *BatchEntryJobDAOInMgo) Find(ctx context.Context, id string) (item *model.BatchEntryJob, err error) {
	query(this.c, func(c *mgoutil.Collection) {
		err = c.Find(bson.M{
			"_id": bson.ObjectIdHex(id),
		}).One(&item)
	})
	return
}

func (this *BatchEntryJobDAOInMgo) Query(ctx context.Context, status enums.BatchEntryJobStatus) (items []*model.BatchEntryJob, err error) {
	query(this.c, func(c *mgoutil.Collection) {
		err = c.Find(bson.M{
			"status": status,
		}).Limit(20).All(&items)
	})
	return
}

func (this *BatchEntryJobDAOInMgo) BatchInsert(ctx context.Context, jobs []*model.BatchEntryJob) (err error) {
	query(this.c, func(c *mgoutil.Collection) {
		docs := make([]interface{}, len(jobs))
		for i, job := range jobs {
			docs[i] = job
		}

		err = c.Insert(docs...)
	})
	return
}

func (this *BatchEntryJobDAOInMgo) StartJob(ctx context.Context, id bson.ObjectId) (err error) {
	query(this.c, func(c *mgoutil.Collection) {
		err = c.Update(bson.M{
			"_id":    id,
			"status": enums.BatchEntryJobStatusNew,
		}, bson.M{
			"$set": bson.M{
				"start_at": time.Now().Unix(),
				"status":   enums.BatchEntryJobStatusProcess,
			},
		})
	})
	return
}

func (this *BatchEntryJobDAOInMgo) UpdateLineNumber(ctx context.Context, id bson.ObjectId, line int64) (err error) {
	query(this.c, func(c *mgoutil.Collection) {
		err = c.Update(bson.M{
			"_id": id,
		}, bson.M{
			"$set": bson.M{
				"line_number": line,
			},
		})
	})
	return
}

func (this *BatchEntryJobDAOInMgo) UpdateStatus(ctx context.Context, id bson.ObjectId, from, to enums.BatchEntryJobStatus) (err error) {
	query(this.c, func(c *mgoutil.Collection) {
		err = c.Update(bson.M{
			"_id":    id,
			"status": from,
		}, bson.M{
			"$set": bson.M{
				"status": to,
			},
		})
	})
	return
}

func (this *BatchEntryJobDAOInMgo) UpdateStatusBySetId(ctx context.Context, setID string, status enums.BatchEntryJobStatus) (err error) {
	query(this.c, func(c *mgoutil.Collection) {
		_, err = c.UpdateAll(bson.M{
			"set_id": setID,
		}, bson.M{
			"$set": bson.M{
				"status": status,
			},
		})
	})
	return
}

func (this *BatchEntryJobDAOInMgo) RemoveBySetId(ctx context.Context, setId string) (err error) {
	query(this.c, func(c *mgoutil.Collection) {
		_, err = c.RemoveAll(bson.M{
			"set_id": setId,
		})
	})
	return
}
