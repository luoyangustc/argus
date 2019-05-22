package job

import (
	"context"
	"sync"

	"gopkg.in/mgo.v2/bson"
	"qiniu.com/argus/censor_private/dao"

	xlog "github.com/qiniu/xlog.v1"
	"qiniu.com/argus/censor_private/proto"
)

type IJob interface {
	Run() error
	Stop() error
	Notify()
}
type Dispatcher struct {
	jobs   map[string]IJob
	init   map[string]struct{} // 记录每个set对应的job是否是第一次启动
	worker IWorker
	sync.Mutex
}

func NewDispatcher(worker IWorker) *Dispatcher {
	return &Dispatcher{
		jobs:   make(map[string]IJob),
		init:   make(map[string]struct{}),
		worker: worker,
	}
}

func (d *Dispatcher) Run(ctx context.Context, set *proto.Set) error {
	xl := xlog.FromContextSafe(ctx)

	d.Lock()
	defer d.Unlock()

	_, ok := d.jobs[set.Id]
	if ok {
		xl.Errorf("set not stopped: %s", set.Id)
		return proto.ErrSetNotStopped
	}

	init := false
	_, ok = d.init[set.Id]
	if !ok {
		init = true
	}

	var job IJob
	switch set.Type {
	case proto.SetTypeMonitorActive:
		job = NewMonitorActive(ctx, set.Id, set.Uri, set.MonitorInterval,
			set.MimeTypes, d.worker, init, d)
	case proto.SetTypeMonitorPassive:
		job = NewProcessJob(ctx, set.Id, set.MimeTypes, d.worker, init)
	case proto.SetTypeTask:
		job = NewProcessTask(ctx, set.Id, set.MimeTypes, d.worker, d)
	default:
	}

	if job != nil {
		err := job.Run()
		if err != nil {
			xl.Errorf("fail to start job (%#v): %v", set, err)
			return err
		}

		d.jobs[set.Id] = job
		d.init[set.Id] = struct{}{}
	}

	return nil
}

func (d *Dispatcher) Stop(ctx context.Context, setId string) error {
	xl := xlog.FromContextSafe(ctx)

	d.Lock()
	defer d.Unlock()

	job, ok := d.jobs[setId]
	if !ok {
		xl.Errorf("set not running: %s", setId)
		return proto.ErrSetNotRunning
	}

	err := job.Stop()
	if err != nil {
		xl.Errorf("fail to stop job (%s): %v", setId, err)
		return err
	}

	delete(d.jobs, setId)
	return nil
}

func (d *Dispatcher) Notify(setId string) {
	d.Lock()
	defer d.Unlock()
	if job, ok := d.jobs[setId]; ok {
		job.Notify()
	}
}

func (d *Dispatcher) Complete(ctx context.Context, setId string) error {
	xl := xlog.FromContextSafe(ctx)

	d.Lock()
	defer d.Unlock()

	_, ok := d.jobs[setId]
	if !ok {
		xl.Errorf("set not running: %s", setId)
		return proto.ErrSetNotRunning
	}
	delete(d.jobs, setId)

	_ = dao.SetDao.Patch(setId, bson.M{"status": proto.SetStatusCompleted})
	return nil
}
