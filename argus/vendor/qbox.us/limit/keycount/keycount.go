package keycount

import (
	"sync"

	"sync/atomic"

	"qbox.us/limit"
)

type keyCountLimit struct {
	mutex   sync.Mutex
	current map[string]uint32
	limit   uint32
}

func New(n int) limit.Limit {
	return &keyCountLimit{current: make(map[string]uint32), limit: uint32(n)}
}

func (l *keyCountLimit) Running() int {

	l.mutex.Lock()
	defer l.mutex.Unlock()

	all := uint32(0)
	for _, v := range l.current {
		all += v
	}
	return int(all)
}

func (l *keyCountLimit) Acquire(key2 []byte) error {

	key := string(key2)

	l.mutex.Lock()
	defer l.mutex.Unlock()

	n := l.current[key]
	if n >= l.limit {
		return limit.ErrLimit
	}
	l.current[key] = n + 1

	return nil
}

func (l *keyCountLimit) Release(key2 []byte) {

	key := string(key2)

	l.mutex.Lock()
	n := l.current[key]
	if n == 1 {
		delete(l.current, key)
	} else {
		l.current[key] = n - 1
	}
	l.mutex.Unlock()
}

// -------------------------------------------------------

type semaphore struct {
	// Number of Acquires - Releases. When this goes to zero, this structure is removed from map.
	// Only updated inside blockingKeyCountLimit.lk lock.
	refs int

	max   int
	value int
	wait  sync.Cond
}

func newSemaphore(max int) *semaphore {
	return &semaphore{
		max:  max,
		wait: sync.Cond{L: new(sync.Mutex)},
	}
}

func (s *semaphore) Running() int {
	s.wait.L.Lock()
	defer s.wait.L.Unlock()

	return int(s.value)
}

func (s *semaphore) Acquire() {
	s.wait.L.Lock()
	defer s.wait.L.Unlock()
	for {
		if s.value+1 <= s.max {
			s.value++
			return
		}
		s.wait.Wait()
	}
	panic("Unexpected branch")
}

func (s *semaphore) Release() {
	s.wait.L.Lock()
	defer s.wait.L.Unlock()
	s.value--
	if s.value < 0 {
		panic("semaphore Release without Acquire")
	}
	s.wait.Signal()
}

type blockingKeyCountLimit struct {
	mutex   sync.RWMutex
	current map[string]*semaphore
	limit   int
}

func NewBlockingKeyCountLimit(n int) limit.Limit {
	return &blockingKeyCountLimit{current: make(map[string]*semaphore), limit: n}
}

func (l *blockingKeyCountLimit) Running() int {
	l.mutex.RLock()
	defer l.mutex.RUnlock()

	all := 0
	for _, v := range l.current {
		all += v.Running()
	}
	return all
}

func (l *blockingKeyCountLimit) Acquire(key2 []byte) error {
	key := string(key2)

	l.mutex.Lock()
	kl, ok := l.current[key]
	if !ok {
		kl = newSemaphore(l.limit)
		l.current[key] = kl

	}
	kl.refs++
	l.mutex.Unlock()

	kl.Acquire()

	return nil
}

func (l *blockingKeyCountLimit) Release(key2 []byte) {
	key := string(key2)

	l.mutex.Lock()
	kl, ok := l.current[key]
	if !ok {
		panic("key not in map. Possible reason: Release without Acquire.")
	}
	kl.refs--
	if kl.refs < 0 {
		panic("internal error: refs < 0")
	}
	if kl.refs == 0 {
		delete(l.current, key)
	}
	l.mutex.Unlock()

	kl.Release()
}

const minusOne = ^uint32(0)

type keySyncLimit struct {
	mutex   sync.RWMutex
	current map[string]*uint32
	limit   uint32
}

func NewSync(n int) limit.StringLimit {
	return &keySyncLimit{current: make(map[string]*uint32), limit: uint32(n)}
}

func (s *keySyncLimit) Running() int {
	s.mutex.RLock()
	defer s.mutex.RUnlock()
	var all uint32
	for _, count := range s.current {
		all += atomic.LoadUint32(count)
	}
	return int(all)
}

func (s *keySyncLimit) Acquire(key string) error {

	s.mutex.RLock()
	count := s.current[key]
	s.mutex.RUnlock()
	if count == nil {
		s.mutex.Lock()
		count = s.current[key]
		if count == nil {
			count = new(uint32)
			s.current[key] = count
		}
		s.mutex.Unlock()
	}
	// 判断和atomic.AddUint32(count, minusOne)之间有一些误差，但是相对而言大多数的限制并不需要那么精确。我觉得使用锁可能会更加影响性能，所以just ignore吧。
	if atomic.AddUint32(count, 1) > s.limit {
		atomic.AddUint32(count, minusOne)
		return limit.ErrLimit
	}
	return nil
}

func (s *keySyncLimit) Release(key string) {
	s.mutex.RLock()
	count := s.current[key]
	s.mutex.RUnlock()
	atomic.AddUint32(count, minusOne)
}
