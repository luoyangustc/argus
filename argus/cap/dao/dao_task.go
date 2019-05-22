package dao

import (
	"context"
	"errors"
	"math"
	"time"

	"qiniu.com/argus/cap/enums"

	mgoutil "github.com/qiniu/db/mgoutil.v3"
	xlog "github.com/qiniu/xlog.v1"
	mgo "gopkg.in/mgo.v2"
	"gopkg.in/mgo.v2/bson"
)

type ITaskDAO interface {
	QueryByID(ctx context.Context, mode, taskId string) (*TaskInMgo, error)
	QueryNotDoneCountByJid(ctx context.Context, mode, jId string) (int, error)
	Insert(ctx context.Context, mode string, tasks ...TaskInMgo) error
	Update(ctx context.Context, mode string, task *TaskInMgo) error
	Remove(ctx context.Context, mode string, taskIds ...string) error

	//处理task的result
	QueryResults(context.Context, string, string, string, int,
	) (string, []TaskInMgo, error)

	//处理auditor处的请求
	FetchTasksForAuditor(ctx context.Context, mode string, packageSize int) ([]TaskInMgo, error)
	CancelTasks(ctx context.Context, mode string, taskIDs []string) error
}

////////////////////////////////////////////////////////////////////////////////

//var _ ITaskDAO = _TaskDAO{}

type _TaskDAO struct {
	//	CapMgoConfig
	colls map[enums.LabelModeType]mgoutil.Collection
}

// NewJobDao New
func NewTaskDao(conf *CapMgoConfig) (ITaskDAO, error) {
	var (
		mgoSessionPoolLimit = 100
		colls               = struct {
			TaskPulp                 mgoutil.Collection `coll:"mode_pulp"`
			TaskPolitician           mgoutil.Collection `coll:"mode_politician"`
			TaskTerror               mgoutil.Collection `coll:"mode_terror"`
			TaskPoliticianPulp       mgoutil.Collection `coll:"mode_politician_pulp"`
			TaskPulpTerror           mgoutil.Collection `coll:"mode_pulp_terror"`
			TaskPoliticianTerror     mgoutil.Collection `coll:"mode_politician_terror"`
			TaskPoliticianPulpTerror mgoutil.Collection `coll:"mode_politician_pulp_terror"`
		}{}
	)

	sess, err := mgoutil.Open(&colls, &conf.Mgo)
	if err != nil {
		return _TaskDAO{}, err
	}
	if conf.MgoPoolLimit > 0 {
		mgoSessionPoolLimit = conf.MgoPoolLimit
	}
	sess.SetPoolLimit(mgoSessionPoolLimit)

	err = colls.TaskPulp.EnsureIndex(mgo.Index{Key: []string{"task_id"}, Unique: true})
	if err != nil {
		return _TaskDAO{}, err
	}

	err = colls.TaskPolitician.EnsureIndex(mgo.Index{Key: []string{"task_id"}, Unique: true})
	if err != nil {
		return _TaskDAO{}, err
	}

	err = colls.TaskTerror.EnsureIndex(mgo.Index{Key: []string{"task_id"}, Unique: true})
	if err != nil {
		return _TaskDAO{}, err
	}

	err = colls.TaskPoliticianPulp.EnsureIndex(mgo.Index{Key: []string{"task_id"}, Unique: true})
	if err != nil {
		return _TaskDAO{}, err
	}

	err = colls.TaskPulpTerror.EnsureIndex(mgo.Index{Key: []string{"task_id"}, Unique: true})
	if err != nil {
		return _TaskDAO{}, err
	}

	err = colls.TaskPoliticianTerror.EnsureIndex(mgo.Index{Key: []string{"task_id"}, Unique: true})
	if err != nil {
		return _TaskDAO{}, err
	}

	err = colls.TaskPoliticianPulpTerror.EnsureIndex(mgo.Index{Key: []string{"task_id"}, Unique: true})
	if err != nil {
		return _TaskDAO{}, err
	}

	taskDao := _TaskDAO{}
	taskDao.colls = make(map[enums.LabelModeType]mgoutil.Collection)
	taskDao.colls[enums.ModePulp] = colls.TaskPulp
	taskDao.colls[enums.ModePolitician] = colls.TaskPolitician
	taskDao.colls[enums.ModeTerror] = colls.TaskTerror
	taskDao.colls[enums.ModePoliticianPulp] = colls.TaskPoliticianPulp
	taskDao.colls[enums.ModePoliticianTerror] = colls.TaskPoliticianTerror
	taskDao.colls[enums.ModePulpTerror] = colls.TaskPulpTerror
	taskDao.colls[enums.ModePoliticianPulpTerror] = colls.TaskPoliticianPulpTerror

	return taskDao, nil
}

func (m _TaskDAO) QueryByID(ctx context.Context, mode, taskId string) (*TaskInMgo, error) {
	var (
		xl   = xlog.FromContextSafe(ctx)
		task TaskInMgo
	)

	sess, ok := m.colls[enums.LabelModeType(mode)]
	if !ok {
		xl.Errorf("get task coll fail, %#v", mode)
		return nil, errors.New("get task coll failed")
	}

	coll := sess.CopySession()
	defer coll.CloseSession()

	err := coll.Find(bson.M{"task_id": taskId}).One(&task)
	if err != nil {
		xl.Warnf("find job failed. %s %v", taskId, err)
	}

	return &task, err
}

func (m _TaskDAO) Insert(ctx context.Context, mode string, tasks ...TaskInMgo) error {
	var (
		xl = xlog.FromContextSafe(ctx)
	)

	sess, ok := m.colls[enums.LabelModeType(mode)]
	if !ok {
		xl.Errorf("get task coll fail, %#v", mode)
		return errors.New("get task coll failed")
	}

	coll := sess.CopySession()
	defer coll.CloseSession()

	for _, task := range tasks {
		task.ID = bson.NewObjectId()
		task.CreateTime = time.Now()
		task.UpdateTime = task.CreateTime
		task.Status = string(enums.TaskTodo)

		err := coll.Insert(task)
		if err != nil {
			//TODO：出错怎么处理
			xl.Errorf("coll.Insert task %s error, %#v", task.TaskID, err.Error())
			return err
		}
	}

	return nil
}

func (m _TaskDAO) Update(ctx context.Context, mode string, task *TaskInMgo) error {
	var (
		xl = xlog.FromContextSafe(ctx)
	)

	sess, ok := m.colls[enums.LabelModeType(mode)]
	if !ok {
		xl.Errorf("get task coll fail, %#v", mode)
		return errors.New("get task coll failed")
	}

	coll := sess.CopySession()
	defer coll.CloseSession()

	task.UpdateTime = time.Now()
	return coll.Update(bson.M{"task_id": task.TaskID}, task)
}

func (m _TaskDAO) Remove(ctx context.Context, mode string, taskIds ...string) error {
	var (
		xl = xlog.FromContextSafe(ctx)
	)

	sess, ok := m.colls[enums.LabelModeType(mode)]
	if !ok {
		xl.Errorf("get task coll fail, %#v", mode)
		return errors.New("get task coll failed")
	}

	coll := sess.CopySession()
	defer coll.CloseSession()

	for _, v := range taskIds {
		err := coll.Remove(bson.M{"task_id": v})
		if err != nil {
			//TODO：出错怎么处理
			xl.Errorf("remove task fail: %#v", v)
			return err
		}
	}

	return nil
}

func (m _TaskDAO) QueryResults(ctx context.Context, mode, jId, marker string, limit int,
) (string, []TaskInMgo, error) {
	var (
		xl = xlog.FromContextSafe(ctx)
	)

	sess, ok := m.colls[enums.LabelModeType(mode)]
	if !ok {
		xl.Errorf("get task coll fail, %#v", mode)
		return "", nil, errors.New("get task coll failed")
	}

	coll := sess.CopySession()
	defer coll.CloseSession()

	var ret = make([]TaskInMgo, 0, limit)

	query := bson.M{"job_id": jId}
	if marker != "" {
		query["task_id"] = bson.M{"$gt": marker}
	}

	q := coll.Find(query).Sort("task_id")
	if limit > 0 {
		q = q.Limit(limit)
	}
	err := q.All(&ret)

	if len(ret) > 0 {
		marker = ret[len(ret)-1].TaskID
	}
	return marker, ret, err
}

func (m _TaskDAO) QueryNotDoneCountByJid(ctx context.Context, mode, jId string) (int, error) {
	var (
		xl = xlog.FromContextSafe(ctx)
	)

	sess, ok := m.colls[enums.LabelModeType(mode)]
	if !ok {
		xl.Errorf("get task coll fail, %#v", mode)
		return math.MaxInt32, errors.New("get task coll failed")
	}

	coll := sess.CopySession()
	defer coll.CloseSession()

	n, err := coll.Find(bson.M{"job_id": jId, "status": bson.M{"$ne": enums.TaskDone}}).Count()
	if err != nil {
		xl.Warnf("find tasks by jid failed. %s %v", jId, err.Error())
		return math.MaxInt32, err
	}

	return n, nil
}

func (m _TaskDAO) FetchTasksForAuditor(ctx context.Context, mode string, packageSize int) ([]TaskInMgo, error) {
	var (
		xl    = xlog.FromContextSafe(ctx)
		tasks []TaskInMgo
	)
	sess, ok := m.colls[enums.LabelModeType(mode)]
	xl.Warnf("========> user mode: %v", enums.LabelModeType(mode))
	if !ok {
		xl.Errorf("get task coll fail, %#v", mode)
		return nil, errors.New("get task coll failed")
	}
	coll := sess.CopySession()
	defer coll.CloseSession()

	err := coll.Find(bson.M{"status": enums.TaskTodo}).Sort("create_time").Limit(packageSize).All(&tasks)
	xl.Infof("mode val: %#v, task Len: %d", mode, len(tasks))

	if err != nil {
		xl.Warnf("find tasks failed. %v", err)
	} else {
		// following code not good, need to optimize
		// change task's status from TODO to DOING
		for _, task := range tasks {
			err = coll.Update(bson.M{"task_id": task.TaskID}, bson.M{"$set": bson.M{"status": enums.TaskDoing}})
			if err != nil {
				xl.Errorf("coll.Update error: %#v", err.Error())
				return nil, err
			}
		}
	}
	return tasks, err
}

func (m _TaskDAO) CancelTasks(ctx context.Context, mode string, taskIDs []string) error {
	var (
		xl = xlog.FromContextSafe(ctx)
	)
	sess, ok := m.colls[enums.LabelModeType(mode)]
	if !ok {
		xl.Errorf("get task coll fail, %#v", mode)
		return errors.New("get task coll failed")
	}
	coll := sess.CopySession()
	defer coll.CloseSession()
	// change task's status from DOING back to TODO
	_, err := coll.UpdateAll(
		bson.M{
			"task_id": bson.M{"$in": taskIDs},
			"status":  enums.TaskDoing,
		}, bson.M{"$set": bson.M{"status": enums.TaskTodo}})
	return err
}
