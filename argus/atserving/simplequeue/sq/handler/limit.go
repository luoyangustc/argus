package handler

import (
	"context"
	"sync"

	"qiniu.com/argus/atserving/simplequeue/sq"
)

var _ sq.Handler = LimitHandler{}

type LimitHandler struct {
	h     sq.Handler
	limit func(sq.Message) bool
}

func NewLimitHandler(
	h sq.Handler,
	limit func(sq.Message) bool,
) LimitHandler {

	return LimitHandler{
		h:     h,
		limit: limit,
	}

}

func (h LimitHandler) HandleMessage(ctx context.Context, m sq.Message) error {

	if h.limit(m) {
		return sq.ErrRequeue
	}
	return h.h.HandleMessage(ctx, m)

}

//----------------------------------------------------------------------------//

var _ sq.Handler = &LimitByConcurrentHandler{}

type LimitByConcurrentHandler struct {
	h sq.Handler

	tag                  sq.MessageTag
	defaultMaxConcurrent int64
	maxConcurrents       map[string]int64

	counts map[string]int64
	*sync.Mutex
}

func NewLimitByConcurrentHandler(
	h sq.Handler,
	tag sq.MessageTag,
	defaultMaxConcurrent int64,
	maxConcurrents map[string]int64,
) *LimitByConcurrentHandler {
	return &LimitByConcurrentHandler{
		h: h,

		tag:                  tag,
		defaultMaxConcurrent: defaultMaxConcurrent,
		maxConcurrents:       maxConcurrents,

		counts: make(map[string]int64),
		Mutex:  new(sync.Mutex),
	}
}

func (h *LimitByConcurrentHandler) maxConcurrent(tag string) int64 {
	max, ok := h.maxConcurrents[tag]
	if ok {
		return max
	}
	return h.defaultMaxConcurrent
}

func (h *LimitByConcurrentHandler) HandleMessage(ctx context.Context, m sq.Message) error {

	var (
		tag           = m.Tag(h.tag)
		maxConcurrent = h.maxConcurrent(tag)
	)

	{ // BEGIN
		h.Lock()
		var count = h.counts[tag]
		if count >= maxConcurrent {
			h.Unlock()
			return sq.ErrRequeue
		}
		h.counts[tag] = count + 1
		h.Unlock()
	}

	defer func() {
		h.Lock()
		defer h.Unlock()
		h.counts[tag] = h.counts[tag] - 1
	}()

	return h.h.HandleMessage(ctx, m)
}
