// +build !python3

package py

import (
	"errors"
	"fmt"
	"sync"
	"sync/atomic"

	python "qiniu.com/argus/com/3rd/github.com/tsavola/go-python"
)

// PyModule ...
type PyModule struct {
	module python.Object
	net    python.Object
}

// NewPyModule ...
func NewPyModule(module string) (*PyModule, error) {
	md, err := python.Import(nil, module)
	if err != nil {
		return nil, fmt.Errorf("PyModule import python module %s failed, error: %s", module, err.Error())
	}
	return &PyModule{module: md}, nil
}

func (m *PyModule) Metrics() (string, error) {
	rest, err := m.module.Call(nil, "metrics")
	if err != nil {
		return "", err
	}
	v, _ := rest.Value(nil)
	return v.(string), nil
}

// NetInit ...
func (m *PyModule) NetInit(function string, args string) error {
	net, err := m.module.Call(nil, function, args)
	if err != nil {
		return fmt.Errorf("Net Loader init net with args(%v) error: %s", args, err.Error())
	}
	e, _, err := net.GetValue(nil, "error")
	if err != nil {
		return fmt.Errorf("Net Loader init net with args(%v) failed, error: %v", args, err)
	}
	eStr, ok := e.(string)
	if !ok {
		return fmt.Errorf("Net Loader init net with args(%v) failed, invalid error result type", args)
	}
	if eStr != "" {
		return fmt.Errorf("Net Loader init net with args(%v) failed, error: %s", args, eStr)
	}
	m.net = net
	return nil
}

// MdRun -  Call python module function
//	args - 	marshaled json string
func (m *PyModule) MdRun(function string, args string) (ret interface{}, err error) {

	if function == "" {
		err = errors.New("PyModule require a function arg at first place")
		return
	}

	ret, err = m.module.CallValue(nil, function, m.net, args)
	if err != nil {
		err = fmt.Errorf("PyModule run function %s(%v) failed, error: %s", function, args, err.Error())
	}

	return
}

// MdRun -  Call python module function
//	args - 	marshaled json string
func (m *PyModule) MdRunWithThreads(function string, args string, threads PyThreads) (ret interface{}, err error) {
	thread := threads.PickOne()
	defer thread.Release()

	if function == "" {
		err = errors.New("PyModule require a function arg at first place")
		return
	}

	ret, err = m.module.CallValue(thread.GetThread(), function, m.net, args)
	if err != nil {
		err = fmt.Errorf("PyModule run function %s(%v) failed, error: %s", function, args, err.Error())
	}

	return
}

func (m *PyModule) MdCall(threads PyThreads, function string, args ...interface{}) (ret interface{}, err error) {
	var thr *python.Thread
	if threads != nil {
		thread := threads.PickOne()
		defer thread.Release()
		thr = thread.GetThread()
	}

	if function == "" {
		err = errors.New("PyModule require a function arg at first place")
		return
	}

	ret, err = m.module.CallValue(thr, function, args...)
	if err != nil {
		err = fmt.Errorf("PyModule run function %s(%v) failed, error: %s", function, args, err.Error())
	}

	return
}

//----------------------------------------------------------------------------//

type PyThreads interface {
	PickOne() _PyThread
}

type _pyThreads struct {
	threads []_PyThread
	*sync.Mutex
}

func NewPyThreads(max_threads int) PyThreads {
	threads := make([]_PyThread, 0, max_threads)
	for i := 0; i < max_threads; i++ {
		threads = append(threads, newPyThread())
	}
	return &_pyThreads{
		threads: threads,
		Mutex:   new(sync.Mutex),
	}
}

func (threads *_pyThreads) PickOne() _PyThread {
	threads.Lock()
	defer threads.Unlock()
	var (
		min   int32 = -1
		index       = -1
	)
	for i, thread := range threads.threads {
		if min == -1 || thread.Count() < min {
			min = thread.Count()
			index = i
		}
	}
	threads.threads[index].Acquire()
	return threads.threads[index]
}

type _PyThread interface {
	Count() int32
	Acquire()
	Release()
	GetThread() *python.Thread
}

type _pyThread struct {
	count int32
	*python.Thread
}

func newPyThread() _PyThread {
	return &_pyThread{
		count:  0,
		Thread: python.NewThread(),
	}
}

func (thread *_pyThread) Count() int32 {
	return atomic.LoadInt32(&thread.count)
}

func (thread *_pyThread) Acquire() {
	atomic.AddInt32(&thread.count, 1)
}

func (thread *_pyThread) Release() {
	atomic.AddInt32(&thread.count, -1)
}

func (thread *_pyThread) GetThread() *python.Thread {
	return thread.Thread
}
