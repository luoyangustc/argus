// +build python3

package py

// PyModule ...
type PyModule struct{}

// NewPyModule ...
func NewPyModule(module string) (*PyModule, error) { return nil, nil }

func (m *PyModule) Metrics() (string, error)                                        { return "", nil }
func (m *PyModule) NetInit(function string, args string) error                      { return nil }
func (m *PyModule) MdRun(function string, args string) (ret interface{}, err error) { return nil, nil }
func (m *PyModule) MdRunWithThreads(function string, args string, threads PyThreads) (ret interface{}, err error) {
	return nil, nil
}
func (m *PyModule) MdCall(threads PyThreads, function string, args ...interface{}) (ret interface{}, err error) {
	return nil, nil
}

//----------------------------------------------------------------------------//

type PyThreads interface {
	PickOne() _PyThread
}

func NewPyThreads(max_threads int) PyThreads { return nil }

type _PyThread interface {
	Count() int32
	Acquire()
	Release()
	GetThread() interface{}
}
