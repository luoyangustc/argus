package job

import (
	"context"
	"sync"

	"github.com/pkg/errors"
	"github.com/qiniu/rpc.v3"
	xlog "github.com/qiniu/xlog.v1"
	"qiniu.com/argus/dbstorage/dao"
	outer "qiniu.com/argus/dbstorage/outer_service"
	"qiniu.com/argus/dbstorage/proto"
	"qiniu.com/argus/dbstorage/source"
)

var (
	defaultNumWorker      = 20
	defaultMaxRunningTask = 1
)

type Dispatcher struct {
	dao             dao.IDao
	workers         []*Worker
	jobPool         chan *FaceJob
	config          *proto.TaskServiceConfig
	runningTasks    map[proto.TaskId]*Distributor
	pendingTasks    []*Distributor
	maxRunningTasks int
	client          *rpc.Client
	sync.Mutex
}

func NewDispatcher(dao dao.IDao, config *proto.TaskServiceConfig) *Dispatcher {
	numWorker := config.ThreadNum
	if numWorker <= 0 {
		numWorker = defaultNumWorker
	}
	jobPool := make(chan *FaceJob, numWorker+10)
	workers := []*Worker{}
	for i := 0; i < numWorker; i++ {
		worker := NewWorker(i, jobPool)
		workers = append(workers, worker)
	}

	maxRunningTasks := config.MaxParallelTaskNum
	if maxRunningTasks <= 0 {
		maxRunningTasks = defaultMaxRunningTask
	}
	d := Dispatcher{
		dao:             dao,
		workers:         workers,
		jobPool:         jobPool,
		config:          config,
		runningTasks:    make(map[proto.TaskId]*Distributor),
		pendingTasks:    make([]*Distributor, 0),
		maxRunningTasks: maxRunningTasks,
		client:          &rpc.DefaultClient,
	}

	//start workers
	for i := 0; i < len(d.workers); i++ {
		d.workers[i].Start()
	}

	return &d
}

func (d *Dispatcher) New(ctx context.Context, task *proto.Task, source source.ISource, faceGroupService outer.IFaceGroup, runInBackground bool, useBar bool) error {
	needUnlock := true
	d.Lock()
	defer func() {
		if needUnlock {
			d.Unlock()
		}
	}()
	xl := xlog.FromContextSafe(ctx)

	//check if already running or pending
	if d.getPendingTaskIndex(task.TaskId) >= 0 || d.isRunning(task.TaskId) {
		return errors.New("task is already started or in pending list")
	}

	distributor := &Distributor{
		ctx:             ctx,
		source:          source,
		runInBackground: runInBackground,
		task:            task,
		Dispatcher:      d,
		faceGroup:       faceGroupService,
	}
	if useBar {
		distributor.bar = InitBar(task.TotalCount)
	}

	//if running list full, add to pending list. otherwise start it
	if len(d.runningTasks) >= d.maxRunningTasks {
		if err := d.updateStatus(ctx, task.TaskId, proto.PENDING); err != nil {
			return err
		}
		d.pendingTasks = append(d.pendingTasks, distributor)
	} else {
		d.runningTasks[task.TaskId] = distributor
		d.Unlock()
		needUnlock = false
		if err := d.start(distributor); err != nil {
			xl.Errorf("fail to start task job: %s", err)
			d.Lock()
			needUnlock = true
			delete(d.runningTasks, task.TaskId)
			return errors.New("fail to start task job")
		}
	}

	return nil
}

func (d *Dispatcher) Stop(ctx context.Context, taskId proto.TaskId) error {
	d.Lock()
	defer d.Unlock()

	//check whether this task exist in pending list
	//if so, remove it from pending list
	if i := d.getPendingTaskIndex(taskId); i >= 0 {
		//if exist, remove it from pending list
		if err := d.updateStatus(ctx, proto.TaskId(taskId), proto.STOPPED); err != nil {
			return err
		}
		d.pendingTasks = append(d.pendingTasks[:i], d.pendingTasks[i+1:]...)
		return nil
	}

	//if not in pending list, try to find it in running list
	if task, ok := d.runningTasks[taskId]; ok {
		//if task is running , send stop signal & update task status to STOPPING
		//status will be set to STOPPED after all the current running job of this task finished,
		if err := d.updateStatus(ctx, proto.TaskId(taskId), proto.STOPPING); err != nil {
			return err
		}
		task.Stop()
	} else {
		return errors.New("fail to stop task, cannot find task job")
	}
	return nil
}

func (d *Dispatcher) Finish(dt *Distributor) {
	needUnlock := true
	d.Lock()
	defer func() {
		if needUnlock {
			d.Unlock()
		}
	}()
	xl := xlog.FromContextSafe(dt.ctx)

	status := proto.COMPLETED
	if dt.cancelled {
		status = proto.STOPPED
	}
	if err := d.updateStatus(dt.ctx, dt.task.TaskId, status); err != nil {
		return
	}
	delete(d.runningTasks, dt.task.TaskId)

	//start the task from pending list if has
	if len(d.runningTasks) < d.maxRunningTasks && len(d.pendingTasks) > 0 {
		tmp := d.pendingTasks[0]
		d.runningTasks[tmp.task.TaskId] = tmp
		d.pendingTasks = d.pendingTasks[1:]
		d.Unlock()
		needUnlock = false
		if err := d.start(tmp); err != nil {
			xl.Errorf("fail to start task from pending list: %s", err)
			return
		}
	}
}

func (d *Dispatcher) start(dt *Distributor) error {
	xl := xlog.FromContextSafe(dt.ctx)

	//init distributor
	if err := dt.Init(); err != nil {
		xl.Errorf("fail to call distributor.Init, going to stop task : %s", err)
		if err2 := d.UpdateLastError(dt.ctx, dt.task.TaskId, proto.TaskError(err.Error())); err2 != nil {
			return err2
		}
		if err2 := d.updateStatus(dt.ctx, dt.task.TaskId, proto.STOPPED); err2 != nil {
			return err2
		}
		return err
	}

	//update task status
	if err := d.updateStatus(dt.ctx, dt.task.TaskId, proto.RUNNING); err != nil {
		return err
	}
	dt.Start()

	return nil
}

func (d *Dispatcher) StopWorkers(ctx context.Context, taskId proto.TaskId) {
	for i := 0; i < len(d.workers); i++ {
		d.workers[i].Stop()
	}
}

func (d *Dispatcher) getPendingTaskIndex(taskId proto.TaskId) int {
	for i, v := range d.pendingTasks {
		if v.task.TaskId == taskId {
			return i
		}
	}
	return -1
}

func (d *Dispatcher) isRunning(taskId proto.TaskId) bool {
	_, ok := d.runningTasks[taskId]
	return ok
}

func (d *Dispatcher) updateStatus(ctx context.Context, taskId proto.TaskId, status proto.TaskStatus) error {
	xl := xlog.FromContextSafe(ctx)
	if err := d.dao.UpdateTaskStatus(ctx, taskId, status); err != nil {
		xl.Errorf("dao.UpdateTaskStatus fail: %s", err)
		return errors.New("fail to update task status")
	}
	return nil
}

func (d *Dispatcher) UpdateLastError(ctx context.Context, taskId proto.TaskId, lastErr proto.TaskError) error {
	xl := xlog.FromContextSafe(ctx)
	if err := d.dao.UpdateTaskError(ctx, taskId, lastErr); err != nil {
		xl.Errorf("dao.UpdateTaskError fail: %s", err)
		return errors.New("fail to update task error")
	}
	return nil
}
