package job

import (
	"context"
	"sort"
	"sync"

	"github.com/pkg/errors"
	httputil "github.com/qiniu/http/httputil.v1"
	xlog "github.com/qiniu/xlog.v1"
	utility "qiniu.com/argus/argus/com/util"
	outer "qiniu.com/argus/dbstorage/outer_service"
	pb "qiniu.com/argus/dbstorage/pb"
	"qiniu.com/argus/dbstorage/proto"
	"qiniu.com/argus/dbstorage/source"
	"qiniu.com/argus/dbstorage/util"
)

type Distributor struct {
	ctx             context.Context
	cancel          context.CancelFunc
	task            *proto.Task
	source          source.ISource
	runInBackground bool
	lastMin         int
	lastMax         int
	lastProcess     []int
	lastErrorCount  int
	process         *proto.SafeArray
	counter         *proto.SafeCounter
	hashDict        *proto.SafeMap
	faceGroup       outer.IFaceGroup
	cancelled       bool
	wg              sync.WaitGroup
	bar             *pb.ProgressBar
	*Dispatcher
	sync.Mutex
}

func (d *Distributor) Init() error {
	d.ctx, d.cancel = context.WithCancel(d.ctx)
	xl := xlog.FromContextSafe(d.ctx)

	//reset last error
	if err := d.UpdateLastError(d.ctx, d.task.TaskId, ""); err != nil {
		return err
	}

	//init params
	d.hashDict = proto.NewSafeMap()
	d.counter = proto.NewSafeCounter()
	d.process = proto.NewSafeArray(d.config.ThreadNum)
	d.lastProcess = make([]int, 0)
	d.lastMin = -1
	d.lastMax = -1
	for i := 0; i < len(d.process.Array); i++ {
		d.process.Array[i] = -1
	}

	//get task log info
	taskLog, err := d.dao.GetTaskLog(d.ctx, d.task.TaskId)
	if err != nil {
		xl.Errorf("dao.GetTaskLog fail: %s", err)
		return errors.New("fail to get task log")
	}
	d.lastErrorCount = taskLog.ErrCount

	//restore the task process if needed
	if taskLog != nil {
		if len(taskLog.Process) > 0 {
			sort.Ints(taskLog.Process)
			d.lastProcess = make([]int, len(taskLog.Process))
			copy(d.lastProcess, taskLog.Process)
			d.lastMin = d.lastProcess[0]
			d.lastMax = d.lastProcess[len(d.lastProcess)-1]
			if len(d.process.Array) < len(taskLog.Process) {
				d.process = proto.NewSafeArray(len(d.process.Array))
			}
			copy(d.process.Array, taskLog.Process)
		}
		if taskLog.Hash != nil {
			for _, v := range taskLog.Hash {
				d.hashDict.Map[v] = struct{}{}
			}
		}
	}

	//create feature group if not exist
	if err := d.faceGroup.CreateGroup(d.ctx, string(d.task.GroupName)); err != nil {
		if !proto.RegGroupExist.MatchString(err.Error()) {
			xl.Errorf("create group <%s> fail: %s", d.task.GroupName, err)
			return errors.Errorf("fail to create group : %s", d.task.GroupName)
		}
	}
	return nil
}

func (d *Distributor) Start() {
	if d.runInBackground {
		go d.start()
	} else {
		d.start()
	}
}

func (d *Distributor) start() {
	defer d.Finish(d)

	xl := xlog.FromContextSafe(d.ctx)
	sourceChan, err := d.source.Read(d.ctx, func(index int) proto.ImageProcess {
		//check whether handled at last time, if so=>skip it
		var process proto.ImageProcess
		if index < d.lastMin {
			process = proto.HANDLED_LAST_TIME
		} else if index <= d.lastMax {
			if util.ArrayContains(d.lastProcess, index) {
				process = proto.HANDLING_LAST_TIME
			} else {
				process = proto.HANDLED_LAST_TIME
			}
		} else {
			process = proto.NOT_HANDLED
		}

		if process == proto.HANDLED_LAST_TIME {
			d.counter.Counter++
			if d.lastErrorCount > 0 {
				IncrementBar(d.bar, false)
				d.lastErrorCount--
			} else {
				IncrementBar(d.bar, true)
			}
		}
		return process
	})
	if err != nil {
		xl.Error("read source failed : ", err)
		d.Stop()
		_ = d.UpdateLastError(d.ctx, d.task.TaskId, proto.TaskError(err.Error()))
		return
	}

	for image := range sourceChan {
		d.wg.Add(1)
		job := NewFaceJob(utility.SpawnContext(d.ctx), image.Index, d, d.faceGroup, image.Content, image.Id, image.URI, image.Tag, image.Desc, image.Process, image.PreCheckErr, &d.wg)
		d.jobPool <- job
	}

	d.wg.Wait()
	CompleteBar(d.bar, "Finished !!!")
}

func (d *Distributor) Stop() {
	d.Lock()
	defer d.Unlock()
	d.cancel()
	d.cancelled = true
}

func (d *Distributor) UpdateProcess(workerIndex int, jobIndex int) error {
	d.process.Lock()
	defer d.process.Unlock()
	d.process.Array[workerIndex] = jobIndex
	return d.dao.UpdateProcess(d.ctx, d.task.TaskId, d.process.Array)
}

func (d *Distributor) UpdateCount() error {
	d.counter.Lock()
	defer d.counter.Unlock()
	d.counter.Counter++
	return d.dao.UpdateTaskCount(d.ctx, d.task.TaskId, d.counter.Counter)
}

func (d *Distributor) UpdateHash(hash string) error {
	return d.dao.UpdateHash(d.ctx, d.task.TaskId, hash)
}

func (d *Distributor) IsHashExist(hash string) bool {
	d.hashDict.Lock()
	defer d.hashDict.Unlock()

	exist := false
	if _, ok := d.hashDict.Map[hash]; ok {
		exist = true
	} else {
		d.hashDict.Map[hash] = struct{}{}
	}
	return exist
}

func (d *Distributor) UpdateErrorLog(uri string, errLog error) {
	xl := xlog.FromContextSafe(d.ctx)
	code := httputil.DetectCode(errLog)
	log := proto.ErrorLog{
		Uri:     uri,
		Code:    proto.ErrorCode(code),
		Message: errLog.Error(),
	}
	if err := d.dao.UpdateErrorLog(d.ctx, d.task.TaskId, log); err != nil {
		xl.Errorf("distributor.UpdateErrorLog fail: %s", err)
	}
}
