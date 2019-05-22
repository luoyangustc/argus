package job

import (
	"context"
	"encoding/hex"
	"fmt"
	"net/http"
	"time"

	"gopkg.in/mgo.v2"
	"gopkg.in/mgo.v2/bson"

	"github.com/qiniu/db/mgoutil.v3"
	"github.com/qiniu/http/httputil.v1"

	. "qiniu.com/argus/bjob/proto"
)

type JobStatus string

const (
	StatusWaiting  JobStatus = "WAITING"
	StatusDoing    JobStatus = "DOING"
	StatusFinished JobStatus = "FINISHED"
)

type Job struct {
	ID string `json:"id"`

	HookURL   string    `json:"hook_url,omitempty"`
	Request   []byte    `json:"request"`
	Env       Env       `json:"env"`
	Status    JobStatus `json:"status"`
	Error     string    `json:"error,omitempty"`
	HookError string    `json:"error_hook,omitempty"`
	Result    []byte    `json:"result,omitempty"`

	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`

	Ref interface{} `json:"-"`
}

type MQs interface {
	GetMQ(context.Context, string) (MQ, error)
}

type MQ interface {
	Submit(context.Context, string, []byte, Env) (string, error)

	Execute(context.Context, int) ([]Job, error)
	Touch(context.Context, *Job) error
	Finish(context.Context, Job) error

	Get(context.Context, string) (Job, error)
	List(context.Context, *JobStatus, *int, *int) ([]Job, error)
}

//----------------------------------------------------------------------------//

type MgoAsMQConfig struct {
	IdleJobTimeout time.Duration  `json:"idle_job_timeout"`
	MgoPoolLimit   int            `json:"mgo_pool_limit"`
	Mgo            mgoutil.Config `json:"mgo"`
}

type _JobInMgo struct {
	ID bson.ObjectId `bson:"_id,omitempty"`

	Cmd string `bson:"cmd"`

	HookURL   string    `json:"hook_url,omitempty"`
	Request   []byte    `bson:"request"`
	Env       Env       `bson:"env"`
	Status    JobStatus `bson:"status"`
	Error     string    `bson:"error,omitempty"`
	HookError string    `bson:"error_hook,omitempty"`
	Result    []byte    `bson:"result,omitempty"`

	CreatedAt time.Time `bson:"created_at"`
	UpdatedAt time.Time `bson:"updated_at"`

	Ver int `bson:"_ver"`
}

func (job _JobInMgo) To() Job {
	return Job{
		ID:        job.ID.Hex(),
		HookURL:   job.HookURL,
		Request:   job.Request,
		Env:       job.Env,
		Status:    job.Status,
		Error:     job.Error,
		HookError: job.HookError,
		Result:    job.Result,
		CreatedAt: job.CreatedAt,
		UpdatedAt: job.UpdatedAt,
		Ref:       job,
	}
}

func (job _JobInMgo) Update(_job *Job) {
	_job.ID = job.ID.Hex()
	_job.HookURL = job.HookURL
	_job.Request = job.Request
	_job.Env = job.Env
	_job.Status = job.Status
	_job.Error = job.Error
	_job.HookError = job.HookError
	_job.Result = job.Result
	_job.CreatedAt = job.CreatedAt
	_job.UpdatedAt = job.UpdatedAt
	_job.Ref = job
}

type _MgoAsMQs struct {
	MgoAsMQConfig

	coll mgoutil.Collection
}

func NewMgoAsMQs(conf MgoAsMQConfig) (_MgoAsMQs, error) {
	coll, err := _MgoAsMQ{}.init(conf)
	if err != nil {
		return _MgoAsMQs{}, err
	}
	return _MgoAsMQs{MgoAsMQConfig: conf, coll: coll}, nil
}

func (ms _MgoAsMQs) GetMQ(ctx context.Context, cmd string) (MQ, error) {
	return _MgoAsMQ{
		MgoAsMQConfig: ms.MgoAsMQConfig,
		coll:          ms.coll,
		cmd:           cmd,
	}, nil
}

type _MgoAsMQ struct {
	MgoAsMQConfig

	coll mgoutil.Collection
	cmd  string
}

func NewMgoAsMQ(conf MgoAsMQConfig, cmd string) (_MgoAsMQ, error) {

	var mq = _MgoAsMQ{MgoAsMQConfig: conf, cmd: cmd}
	var err error
	mq.coll, err = mq.init(conf)
	if err != nil {
		return _MgoAsMQ{}, err
	}
	return mq, nil
}

func (m _MgoAsMQ) init(conf MgoAsMQConfig) (mgoutil.Collection, error) {
	var (
		mgoSessionPoolLimit = 100
		colls               = struct {
			Jobs mgoutil.Collection `coll:"jobs"`
		}{}
	)
	sess, err := mgoutil.Open(&colls, &conf.Mgo)
	if err != nil {
		return mgoutil.Collection{}, err
	}
	if conf.MgoPoolLimit > 0 {
		mgoSessionPoolLimit = conf.MgoPoolLimit
	}
	sess.SetPoolLimit(mgoSessionPoolLimit)

	if err := colls.Jobs.EnsureIndex(mgo.Index{Key: []string{"cmd"}}); err != nil {
		panic(err)
	}
	if err := colls.Jobs.EnsureIndex(mgo.Index{Key: []string{"status"}}); err != nil {
		panic(err)
	}
	if err := colls.Jobs.EnsureIndex(mgo.Index{Key: []string{"created_at"}}); err != nil {
		panic(err)
	}
	if err := colls.Jobs.EnsureIndex(mgo.Index{Key: []string{"updated_at"}}); err != nil {
		panic(err)
	}

	return colls.Jobs, nil
}

func (m _MgoAsMQ) Submit(ctx context.Context, hookURL string, req []byte, env Env) (string, error) {
	var (
		job = _JobInMgo{
			ID:        bson.NewObjectId(),
			Cmd:       m.cmd,
			HookURL:   hookURL,
			Request:   req,
			Env:       env,
			Status:    StatusWaiting,
			CreatedAt: time.Now(),
			UpdatedAt: time.Now(),
			Ver:       0,
		}
	)

	coll := m.coll.CopySession()
	defer coll.CloseSession()

	return job.ID.Hex(), coll.Insert(job)
}

func (m _MgoAsMQ) update(
	ctx context.Context,
	coll mgoutil.Collection,
	job *_JobInMgo,
) error {
	job.Ver = job.Ver + 1
	job.UpdatedAt = time.Now()
	// TODO 细分错误
	return coll.Update(
		bson.M{"_id": job.ID, "_ver": job.Ver - 1},
		job,
	)
}

func (m _MgoAsMQ) Execute(ctx context.Context, limit int) ([]Job, error) {
	coll := m.coll.CopySession()
	defer coll.CloseSession()

	var (
		jobs = make([]Job, 0, limit)
		now  = time.Now()
	)

	for {
		var (
			_len  = len(jobs)
			_jobs = make([]_JobInMgo, 0)
		)
		if _len >= limit {
			break
		}
		if m.IdleJobTimeout > 0 {
			err := coll.
				Find(
					bson.M{
						"cmd":        m.cmd,
						"status":     StatusDoing,
						"updated_at": bson.M{"$lt": now.Add(-1 * m.IdleJobTimeout)},
					}).
				Limit(limit - _len).
				All(&_jobs)
			if err != nil {
				return jobs, err
			}
		}
		if len(_jobs) == 0 {
			err := coll.
				Find(bson.M{"cmd": m.cmd, "status": StatusWaiting}).
				Limit(limit - _len).
				All(&_jobs)
			if err != nil {
				return jobs, err
			}
		}
		if len(_jobs) == 0 {
			break
		}
		for _, _job := range _jobs {
			_job.Status = StatusDoing
			err := m.update(ctx, coll, &_job)
			if err == mgo.ErrNotFound {
				continue
			}
			if err != nil {
				return jobs, err
			}
			jobs = append(jobs, _job.To())
		}
	}

	return jobs, nil
}

func (m _MgoAsMQ) Touch(ctx context.Context, job *Job) error {
	coll := m.coll.CopySession()
	defer coll.CloseSession()

	var (
		_job = job.Ref.(_JobInMgo)
	)
	err := m.update(ctx, coll, &_job)
	if err != nil {
		return err
	}
	_job.Update(job)
	return nil
}

func (m _MgoAsMQ) Finish(ctx context.Context, job Job) error {
	coll := m.coll.CopySession()
	defer coll.CloseSession()

	var (
		_job = job.Ref.(_JobInMgo)
	)
	_job.Status = StatusFinished
	if job.Error == "" {
		_job.Result = job.Result
	} else {
		_job.Error = job.Error
	}
	err := m.update(ctx, coll, &_job)
	if err != nil {
		return err
	}
	return nil
}

func (m _MgoAsMQ) Get(ctx context.Context, id string) (Job, error) {
	coll := m.coll.CopySession()
	defer coll.CloseSession()

	var (
		_job = _JobInMgo{}
	)

	d, err := hex.DecodeString(id)
	if err != nil || len(d) != 12 {
		return Job{}, httputil.NewError(http.StatusNotFound, fmt.Sprintf("invalid input Job id: %q", id))
	}

	err = coll.Find(bson.M{"_id": bson.ObjectId(d)}).One(&_job)
	if err != nil {
		return Job{}, err
	}
	return _job.To(), nil
}

func (m _MgoAsMQ) List(
	ctx context.Context,
	status *JobStatus,
	offset, limit *int,
) ([]Job, error) {
	coll := m.coll.CopySession()
	defer coll.CloseSession()

	var (
		_jobs  = make([]_JobInMgo, 0)
		_query = bson.M{}
	)
	if status != nil {
		_query["status"] = *status
	}
	query := coll.Find(_query)
	if offset != nil {
		query = query.Skip(*offset)
	}
	if limit != nil {
		query = query.Limit(*limit)
	}
	err := query.All(&_jobs)
	if err != nil {
		return nil, err
	}
	var jobs = make([]Job, 0, len(_jobs))
	for _, _job := range _jobs {
		jobs = append(jobs, _job.To())
	}
	return jobs, nil
}
