package dao

import (
	"context"
	"time"

	"qiniu.com/argus/cap/enums"

	"gopkg.in/mgo.v2"
	"gopkg.in/mgo.v2/bson"

	"github.com/qiniu/db/mgoutil.v3"
	"github.com/qiniu/xlog.v1"
)

type IJobDAO interface {
	QueryByID(ctx context.Context, jobID string) (*JobInMgo, error)
	Insert(ctx context.Context, jobs ...JobInMgo) error
	Update(ctx context.Context, job *JobInMgo) error
	Remove(ctx context.Context, jobID string) error
}

////////////////////////////////////////////////////////////////////////////////

var _ IJobDAO = _JobDAO{}

type _JobDAO struct {
	//	CapMgoConfig
	coll mgoutil.Collection
}

// NewJobDao New
func NewJobDao(conf *CapMgoConfig) (IJobDAO, error) {
	var (
		mgoSessionPoolLimit = 100
		colls               = struct {
			Set mgoutil.Collection `coll:"job"`
		}{}
	)
	sess, err := mgoutil.Open(&colls, &conf.Mgo)
	if err != nil {
		return _JobDAO{}, err
	}
	if conf.MgoPoolLimit > 0 {
		mgoSessionPoolLimit = conf.MgoPoolLimit
	}
	sess.SetPoolLimit(mgoSessionPoolLimit)

	err = colls.Set.EnsureIndex(mgo.Index{Key: []string{"job_id"}, Unique: true})
	if err != nil {
		return _JobDAO{}, err
	}

	return _JobDAO{coll: colls.Set}, nil
}

func (m _JobDAO) QueryByID(ctx context.Context, jobID string) (*JobInMgo, error) {
	coll := m.coll.CopySession()
	defer coll.CloseSession()

	var (
		xl  = xlog.FromContextSafe(ctx)
		job JobInMgo
	)

	err := coll.Find(bson.M{"job_id": jobID}).One(&job)
	if err != nil {
		xl.Warnf("find job failed. %s %v", jobID, err)
	}

	return &job, err
}

func (m _JobDAO) Insert(ctx context.Context, jobs ...JobInMgo) error {
	coll := m.coll.CopySession()
	defer coll.CloseSession()

	xl := xlog.FromContextSafe(ctx)
	xl.Infof("jobs: %#v", jobs[0])
	for _, job := range jobs {

		job.ID = bson.NewObjectId()
		job.CreateTime = time.Now()
		job.Status = string(enums.JobBegin)

		err := coll.Insert(job)
		if err != nil {
			return err
		}
	}

	return nil
}

func (m _JobDAO) Update(ctx context.Context, job *JobInMgo) error {
	coll := m.coll.CopySession()
	defer coll.CloseSession()

	return coll.Update(bson.M{"job_id": job.JobID}, job)
}

func (m _JobDAO) Remove(ctx context.Context, jobID string) error {
	coll := m.coll.CopySession()
	defer coll.CloseSession()

	return coll.Remove(bson.M{"job_id": jobID})
}
