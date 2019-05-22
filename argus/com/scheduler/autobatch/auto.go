package autobatch

import (
	"container/list"
	"context"
	"sync"
	"sync/atomic"
)

type BatchConfig struct {
	// MaxWaitingMS     time.Duration `json:"max_waiting_ms"`
	MaxBatchSize int `json:"max_batch_size"`
	// MaxWaitingLength int           `json:"max_waiting_length"`
	MaxConcurrent int `json:"max_concurrent"`
}

type BatchFunc func(context.Context, []interface{}) ([]interface{}, error)

type _Task struct {
	ID        int64
	Cancelled uint32
	Params    interface{}
	Result    interface{}
	ch        chan bool
}

func newTask(id int64, params interface{}) *_Task {
	return &_Task{ID: id, Params: params, ch: make(chan bool, 1)}
}
func (t *_Task) Cancel()           { atomic.StoreUint32(&t.Cancelled, 1) }
func (t *_Task) IsCancelled() bool { return atomic.LoadUint32(&t.Cancelled) > 0 }
func (t *_Task) Done(ret interface{}) {
	t.Result = ret
	close(t.ch)
}
func (t *_Task) Watch() <-chan bool { return t.ch }
func (t *_Task) GetResult() (interface{}, error) {
	if t.Result == nil {
		return nil, nil
	}
	if err, ok := t.Result.(error); ok {
		return nil, err
	}
	return t.Result, nil
}

type _Tasks struct {
	*list.List
}

func newTasks() _Tasks { return _Tasks{List: list.New()} }

func (ts _Tasks) Push(task *_Task) { ts.List.PushBack(task) }
func (ts _Tasks) Has(task *_Task) bool {
	for e := ts.List.Front(); e != nil; e = e.Next() {
		if e.Value == nil {
			continue
		}
		_task := e.Value.(*_Task)
		if _task.ID == task.ID {
			return true
		}
	}
	return false
}

func (ts _Tasks) Pick(task *_Task) bool {
	for e := ts.List.Front(); e != nil; e = e.Next() {
		if e.Value == nil {
			continue
		}
		_task := e.Value.(*_Task)
		if _task.ID == task.ID {
			e.Value = nil
			return true
		}
	}
	return false
}

func (ts _Tasks) PopN(n int, tasks []*_Task) []*_Task {
	var m = len(tasks)
	for len(tasks)-m < n {
		var find = false
		for e := ts.List.Front(); e != nil; {
			e0 := e
			e = e.Next()
			ts.List.Remove(e0)
			if e0.Value == nil {
				continue
			}
			_task := e0.Value.(*_Task)
			if _task.IsCancelled() {
				continue
			}
			tasks = append(tasks, _task)
			find = true
			break
		}
		if !find {
			break
		}
	}
	return tasks
}

type Batch struct {
	BatchConfig
	Func BatchFunc

	lastID int64
	tasks  _Tasks
	sync.Mutex

	running chan bool
}

func NewBatch(conf BatchConfig, batchFunc BatchFunc) *Batch {
	if conf.MaxConcurrent == 0 {
		conf.MaxConcurrent = 1
	}
	running := make(chan bool, conf.MaxConcurrent)
	for i := 0; i < conf.MaxConcurrent; i++ {
		running <- true
	}
	return &Batch{BatchConfig: conf, Func: batchFunc, tasks: newTasks(), running: running}
}

func (b *Batch) Do(ctx context.Context, params interface{}) (interface{}, error) {

	task := newTask(atomic.AddInt64(&b.lastID, 1), params)
	b.Lock()
	b.tasks.Push(task)
	b.Unlock()

	go b.tryRunOneBatch(ctx, task)

	// select {
	// case <-time.After(b.MaxWaitingMS * time.Millisecond):
	// 	go b.tryRunOneBatch(ctx, task)
	// case <-task.Watch():
	// 	ret, err := task.GetResult()
	// 	return ret, err
	// case <-ctx.Done():
	// 	task.Cancel()
	// 	return nil, ctx.Err()
	// }

	select {
	case <-task.Watch():
		ret, err := task.GetResult()
		return ret, err
	case <-ctx.Done():
		task.Cancel()
		return nil, ctx.Err()
	}
}

func (b *Batch) tryRunOneBatch(ctx context.Context, task *_Task) {
	if task != nil {
		b.Lock()
		ok := b.tasks.Has(task)
		b.Unlock()
		if !ok {
			return
		}
	}

	<-b.running
	defer func() { b.running <- true }()

	var tasks = make([]*_Task, 0, b.MaxBatchSize)

	b.Lock()
	if task != nil {
		if !b.tasks.Pick(task) {
			b.Unlock()
			return
		}
		tasks = append(tasks, task)
	}
	tasks = b.tasks.PopN(b.MaxBatchSize-len(tasks), tasks)
	b.Unlock()

	b.runOneBatch(ctx, tasks)
}

func (b *Batch) runOneBatch(ctx context.Context, tasks []*_Task) {
	var params = make([]interface{}, len(tasks))
	for i, task := range tasks {
		params[i] = task.Params
	}
	rets, err := b.Func(ctx, params)
	for i, task := range tasks {
		if err != nil {
			task.Done(err)
		} else {
			task.Done(rets[i])
		}
	}
}
