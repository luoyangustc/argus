package broker

import (
	"math/rand"
	"sync"
	"sync/atomic"
)

type lb interface {
	selectNextHostIndex() uint32
	selectHostOfQueueAndOp(queue string, op int) uint32
	host(i uint32) string
}

type roundrobin struct {
	hosts     []string
	nextIndex uint32

	rrOfQueues [opCount]nexter
}

func (rr *roundrobin) selectNextHostIndex() uint32 {
	return atomic.AddUint32(&rr.nextIndex, 1)
}

func (rr *roundrobin) host(i uint32) string {
	return rr.hosts[i%uint32(len(rr.hosts))]
}

func (rr *roundrobin) selectHostOfQueueAndOp(queue string, op int) uint32 {
	return rr.rrOfQueues[op].next(queue)
}

func newRoundrobin(hosts []string) lb {
	rr := &roundrobin{
		hosts: hosts,
	}

	for i := range rr.rrOfQueues {
		rr.rrOfQueues[i] = &roundrobinForQueue{
			queueIndex: make(map[string]int32, 128),
			indexes:    make([]uint32, 0, 128),
		}
	}

	rr.nextIndex = uint32(rand.Intn(len(hosts)))

	return rr
}

type nexter interface {
	next(string) uint32
}

type roundrobinForQueue struct {
	sync.RWMutex

	queueIndex map[string]int32
	indexes    []uint32
}

func (rr *roundrobinForQueue) next(queue string) uint32 {
	rr.RLock()
	i, ok := rr.queueIndex[queue]
	rr.RUnlock()

	if !ok {
		rr.Lock()
		i, ok = rr.queueIndex[queue]
		if !ok {
			i = int32(len(rr.indexes))
			rr.queueIndex[queue] = i
			rr.indexes = append(rr.indexes, 0)
		}
		rr.Unlock()
	}

	return atomic.AddUint32(&rr.indexes[i], 1)
}
