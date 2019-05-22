package main

import (
	"sync"
)

type subprocessStatus struct {
	forwardStartSuccessNum   int
	inferenceStartSuccessNum int
	m                        sync.Mutex
	ch                       chan bool
	once                     sync.Once
}

func (s *subprocessStatus) close() {
	close(s.ch)
	xl.Info("at least one forward and one inference start ok")
}

func (s *subprocessStatus) forwardStartSuccess() {
	s.m.Lock()
	s.forwardStartSuccessNum++
	if s.forwardStartSuccessNum > 0 && s.inferenceStartSuccessNum > 0 {
		s.once.Do(s.close)
	}
	s.m.Unlock()
}

func (s *subprocessStatus) inferenceStartSuccess() {
	s.m.Lock()
	s.inferenceStartSuccessNum++
	if s.forwardStartSuccessNum > 0 && s.inferenceStartSuccessNum > 0 {
		s.once.Do(s.close)
	}
	s.m.Unlock()
}

func (s *subprocessStatus) waitStart() {
	<-s.ch
}

var globalSubprocessStatus = subprocessStatus{ch: make(chan bool)}
