package handler

import (
	"context"
	"sync"
	"time"

	"qiniu.com/argus/atserving/simplequeue/sq"
)

type StatsConfig struct {
	Tag      sq.MessageTag
	Duration time.Duration
}

func NewStatsConfig(
	tag sq.MessageTag,
	duration time.Duration,
) StatsConfig {
	return StatsConfig{
		Tag:      tag,
		Duration: duration,
	}
}

//----------------------------------------------------------------------------//

var _ sq.Handler = &StatsHandler{}

type StatsHandler struct {
	h            sq.Handler
	config       StatsConfig
	tickDuration time.Duration

	index      uint64
	indexMutex *sync.Mutex

	m map[string]map[uint64]time.Time
	*sync.RWMutex
}

func NewStatsHandler(
	h sq.Handler,
	config StatsConfig,
	tickDuration time.Duration,
) *StatsHandler {

	return &StatsHandler{
		h:            h,
		config:       config,
		tickDuration: tickDuration,

		index:      0,
		indexMutex: new(sync.Mutex),

		m:       make(map[string]map[uint64]time.Time),
		RWMutex: new(sync.RWMutex),
	}
}

func (h *StatsHandler) Stats() map[string]time.Duration {

	var (
		m   = make(map[string]time.Duration)
		now = time.Now()
	)

	h.RLock()
	defer h.RUnlock()

	for tag, m1 := range h.m {
		var sum time.Duration
		for _, begin := range m1 {
			sum += now.Sub(begin)
		}
		m[tag] = sum
	}

	return m
}

const _MAXID uint64 = 1 << 63

func (h *StatsHandler) newID() uint64 {
	h.indexMutex.Lock()
	defer h.indexMutex.Unlock()

	h.index += 1
	if h.index >= _MAXID {
		h.index = 0
	}
	return h.index
}

func (h *StatsHandler) HandleMessage(ctx context.Context, m sq.Message) error {

	var (
		tag, id = m.Tag(h.config.Tag), h.newID()
		now     = time.Now()
	)

	{ // BEGIN
		h.Lock()
		m1, ok := h.m[tag]
		if !ok {
			m1 = make(map[uint64]time.Time)
		}
		m1[id] = now // check ?
		h.m[tag] = m1
		h.Unlock()
	}

	defer func() {
		h.Lock()
		defer h.Unlock()
		m1 := h.m[tag]
		delete(m1, id)
	}()

	return h.h.HandleMessage(ctx, m)

}
