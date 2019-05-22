package handler

import (
	"context"
	"sync/atomic"
	"time"

	"qiniu.com/argus/atserving/simplequeue/sq"
)

var _ sq.Handler = DelayHandler{}

type DelayHandler struct {
	h      sq.Handler
	isWait func() bool
}

func NewDelayHandler(h sq.Handler, isWait func() bool) DelayHandler {
	return DelayHandler{
		h:      h,
		isWait: isWait,
	}
}

func (h DelayHandler) HandleMessage(ctx context.Context, m sq.Message) error {

	if h.isWait != nil && h.isWait() {
		return sq.ErrRequeueWithoutBackoff
	}

	return h.h.HandleMessage(ctx, m)
}

//----------------------------------------------------------------------------//

func AfterOtherQueues(
	lastModifieds []*int64,
	duration time.Duration,
) func() bool {

	return func() bool {
		now := time.Now().UnixNano()
		for _, lastModified := range lastModifieds {
			if now-atomic.LoadInt64(lastModified) < duration.Nanoseconds() {
				return true
			}
		}
		return false
	}
}
