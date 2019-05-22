package dao

import (
	"context"
	"io/ioutil"

	"github.com/pkg/errors"
	"github.com/qiniu/db/mgoutil.v3"
	mgo "gopkg.in/mgo.v2"
	"gopkg.in/mgo.v2/bson"

	"qiniu.com/argus/dbstorage/proto"
)

const (
	defaultCollSessionPoolLimit = 100
)

type _MgoCollections struct {
	Tasks mgoutil.Collection `coll:"tasks"`
	Logs  mgoutil.Collection `coll:"logs"`
}

var _ IDao = new(DbDao)

type DbDao struct {
	tasksColl *mgoutil.Collection
	logsColl  *mgoutil.Collection
}

func NewDbDao(cfg *mgoutil.Config) (IDao, error) {
	collections := _MgoCollections{}
	mgoSession, err := mgoutil.Open(&collections, cfg)
	if err != nil {
		return nil, err
	}

	mgoSession.SetPoolLimit(defaultCollSessionPoolLimit)

	d := &DbDao{
		tasksColl: &collections.Tasks,
		logsColl:  &collections.Logs}
	return d, nil
}

func (d *DbDao) NewTask(ctx context.Context, taskId proto.TaskId, groupName proto.GroupName, taskConfig proto.TaskConfig, count int, ext string, uid uint32) error {
	col := d.tasksColl.CopySession()
	defer col.CloseSession()

	if err := col.Insert(proto.Task{
		TaskId:     taskId,
		Uid:        uid,
		GroupName:  groupName,
		Config:     taskConfig,
		FileName:   string(taskId),
		FileExt:    ext,
		TotalCount: count,
		Status:     proto.CREATED,
	}); err != nil {
		return errors.Errorf("fail to add task due to db err: %s", err)
	}

	return nil
}

func (d *DbDao) DeleteTask(ctx context.Context, taskId proto.TaskId, uid uint32) error {
	col := d.tasksColl.CopySession()
	defer col.CloseSession()

	if _, err := col.RemoveAll(bson.M{"uid": uid, "task_id": taskId}); err != nil {
		return errors.Errorf("fail to delete task due to db err: %s", err)
	}
	return nil
}

func (d *DbDao) GetTask(ctx context.Context, taskId proto.TaskId, uid uint32) (*proto.Task, error) {
	col := d.tasksColl.CopySession()
	defer col.CloseSession()

	var task proto.Task
	if err := col.Find(bson.M{"uid": uid, "task_id": taskId}).One(&task); err != nil {
		if err != mgo.ErrNotFound {
			return nil, errors.Errorf("fail to get task due to db err: %s", err)
		}
		return nil, errors.New("task does not exist")
	}
	return &task, nil
}

func (d *DbDao) GetTaskList(ctx context.Context, groupName proto.GroupName, status proto.TaskStatus, uid uint32) ([]proto.Task, error) {
	col := d.tasksColl.CopySession()
	defer col.CloseSession()

	param := bson.M{"uid": uid, "group_name": groupName}
	if status != "" {
		param["status"] = status
	}

	var tasks []proto.Task
	if err := col.Find(param).Select(bson.M{"task_id": 1, "status": 1}).All(&tasks); err != nil {
		return nil, errors.Errorf("fail to get task ids due to db err: %s", err)
	}

	return tasks, nil
}

func (d *DbDao) ResetTask(ctx context.Context, status proto.TaskStatus, condition ...proto.TaskStatus) error {
	col := d.tasksColl.CopySession()
	defer col.CloseSession()

	if _, err := col.UpdateAll(bson.M{"status": bson.M{"$in": condition}},
		bson.M{"$set": bson.M{"status": status}}); err != nil {
		return errors.Errorf("fail to reset task due to db err: %s", err)
	}
	return nil
}

func (d *DbDao) UpdateTaskStatus(ctx context.Context, taskId proto.TaskId, status proto.TaskStatus) error {
	col := d.tasksColl.CopySession()
	defer col.CloseSession()

	if err := col.Update(bson.M{"task_id": taskId},
		bson.M{"$set": bson.M{"status": status}}); err != nil {
		if err != mgo.ErrNotFound {
			return errors.Errorf("fail to update task status due to db err: %s", err)
		}
		return errors.New("task does not exist")
	}

	return nil
}

func (d *DbDao) UpdateTaskCount(ctx context.Context, taskId proto.TaskId, count int) error {
	col := d.tasksColl.CopySession()
	defer col.CloseSession()

	if err := col.Update(bson.M{"task_id": taskId},
		bson.M{"$set": bson.M{"handled_count": count}}); err != nil {
		if err != mgo.ErrNotFound {
			return errors.Errorf("fail to update task count due to db err: %s", err)
		}
		return errors.New("task does not exist")
	}

	return nil
}

func (d *DbDao) UpdateTaskError(ctx context.Context, taskId proto.TaskId, err proto.TaskError) error {
	col := d.tasksColl.CopySession()
	defer col.CloseSession()

	if err := col.Update(bson.M{"task_id": taskId},
		bson.M{"$set": bson.M{"last_error": err}}); err != nil {
		if err != mgo.ErrNotFound {
			return errors.Errorf("fail to update task last_error due to db err: %s", err)
		}
		return errors.New("task does not exist")
	}

	return nil
}

func (d *DbDao) NewTaskFile(ctx context.Context, fileName string, fileContent []byte) error {
	col := d.tasksColl.CopySession()
	defer col.CloseSession()
	file, err := col.Database.GridFS("fs").Create(fileName)
	if err != nil {
		return errors.Errorf("fail to add task file due to create gridfs err: %s", err)
	}
	defer file.Close()
	file.Write(fileContent)
	return nil
}

func (d *DbDao) DeleteTaskFile(ctx context.Context, fileName string) error {
	col := d.tasksColl.CopySession()
	defer col.CloseSession()
	err := col.Database.GridFS("fs").Remove(fileName)
	if err != nil {
		return errors.Errorf("fail to delete task file due to remove gridfs err: %s", err)
	}
	return nil
}

func (d *DbDao) GetTaskFile(ctx context.Context, fileName string) ([]byte, error) {
	col := d.tasksColl.CopySession()
	defer col.CloseSession()
	file, err := col.Database.GridFS("fs").Open(fileName)
	if err != nil {
		return nil, errors.Errorf("fail to open task file due to open gridfs err: %s", err)
	}
	defer file.Close()
	content, err := ioutil.ReadAll(file)
	if err != nil {
		return nil, errors.Errorf("fail to read task file due to read gridfs err: %s", err)
	}
	return content, nil
}

func (d *DbDao) NewTaskLog(ctx context.Context, taskId proto.TaskId, uid uint32) error {
	col := d.logsColl.CopySession()
	defer col.CloseSession()

	if err := col.Insert(proto.Log{Uid: uid, TaskId: taskId}); err != nil {
		return errors.Errorf("fail to add task log due to db err: %s", err)
	}
	return nil
}

func (d *DbDao) GetTaskLog(ctx context.Context, taskId proto.TaskId) (*proto.Log, error) {
	col := d.logsColl.CopySession()
	defer col.CloseSession()

	var log proto.Log
	if err := col.Find(bson.M{"task_id": taskId}).Select(bson.M{"process": 1, "hash": 1, "error_count": 1}).One(&log); err != nil {
		if err != mgo.ErrNotFound {
			return nil, errors.Errorf("fail to get task log due to db err: %s", err)
		}
		return nil, errors.New("task log does not exist")
	}

	return &log, nil
}

func (d *DbDao) DeleteTaskLog(ctx context.Context, taskId proto.TaskId) error {
	col := d.logsColl.CopySession()
	defer col.CloseSession()

	if _, err := col.RemoveAll(bson.M{"task_id": taskId}); err != nil {
		return errors.Errorf("fail to delete task log due to db err: %s", err)
	}

	return nil
}

func (d *DbDao) UpdateProcess(ctx context.Context, taskId proto.TaskId, process []int) error {
	col := d.logsColl.CopySession()
	defer col.CloseSession()

	if err := col.Update(bson.M{"task_id": taskId},
		bson.M{"$set": bson.M{"process": process}}); err != nil {
		if err != mgo.ErrNotFound {
			return errors.Errorf("fail to update task process due to db err: %s", err)
		}
		return errors.New("task log does not exist")
	}
	return nil
}

func (d *DbDao) UpdateHash(ctx context.Context, taskId proto.TaskId, hash string) error {
	col := d.logsColl.CopySession()
	defer col.CloseSession()

	if err := col.Update(bson.M{"task_id": taskId},
		bson.M{"$push": bson.M{"hash": hash}}); err != nil {
		if err != mgo.ErrNotFound {
			return errors.Errorf("fail to update task hash due to db err: %s", err)
		}
		return errors.New("task log does not exist")
	}
	return nil
}

func (d *DbDao) UpdateErrorLog(ctx context.Context, taskId proto.TaskId, errLog proto.ErrorLog) error {
	col := d.logsColl.CopySession()
	defer col.CloseSession()

	if err := col.Update(bson.M{"task_id": taskId},
		bson.M{"$push": bson.M{"error": errLog},
			"$inc": bson.M{"error_count": 1}}); err != nil {
		if err != mgo.ErrNotFound {
			return errors.Errorf("fail to update error log due to db err: %s", err)
		}
		return errors.New("task log does not exist")
	}
	return nil
}

func (d *DbDao) GetErrorLog(ctx context.Context, taskId proto.TaskId, skip, limit int, uid uint32) ([]proto.ErrorLog, int, error) {
	col := d.logsColl.CopySession()
	defer col.CloseSession()

	param := bson.M{"hash": 0, "process": 0}
	if limit != 0 {
		// limit!=0 => get part of log
		param["error"] = bson.M{"$slice": []int{skip, limit}}
	}

	var log proto.Log
	if err := col.Find(bson.M{"uid": uid, "task_id": taskId}).Select(param).One(&log); err != nil {
		if err != mgo.ErrNotFound {
			return nil, 0, errors.Errorf("fail to get error log due to db err: %s", err)
		}
		return nil, 0, errors.New("task log does not exist")
	}
	return log.Err, log.ErrCount, nil
}

func (d *DbDao) GetErrorCount(ctx context.Context, taskId proto.TaskId, uid uint32) (int, error) {
	col := d.logsColl.CopySession()
	defer col.CloseSession()

	var log proto.Log
	if err := col.Find(bson.M{"uid": uid, "task_id": taskId}).Select(bson.M{"error_count": 1}).One(&log); err != nil {
		if err != mgo.ErrNotFound {
			return 0, errors.Errorf("fail to get error count due to db err: %s", err)
		}
		return 0, errors.New("task log does not exist")
	}
	return log.ErrCount, nil
}
