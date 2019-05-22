package msq

import (
	"context"
	"sync/atomic"
	"time"

	"qiniu.com/argus/atserving/simplequeue/sq"
	qHandler "qiniu.com/argus/atserving/simplequeue/sq/handler"
)

var (
	Default_WaitDuration time.Duration = time.Millisecond * 50
)

type MultiConsumer struct {
	cs     []*sq.Consumer
	counts []int64
}

func NewMultiConsumer(
	configs []sq.ConsumerConfig,
	handlers []sq.Handler,
	newMessage func(sq.Message) sq.Message,
) (*MultiConsumer, error) {
	var (
		err error

		counts        []int64 = make([]int64, len(configs))
		lastModifieds []int64 = make([]int64, len(configs))

		cs = make([]*sq.Consumer, len(configs))

		newHandler func(*int64, *int64, sq.Handler) sq.Handler = func(count, lastModified *int64, handler sq.Handler) sq.Handler {
			return sq.HandlerFunc(func(ctx context.Context, m sq.Message) error {
				atomic.AddInt64(count, 1)
				atomic.StoreInt64(lastModified, time.Now().UnixNano())
				return handler.HandleMessage(ctx, m)
			})
		}
		newWaitFunc func(int) func() bool = func(n int) func() bool {
			if n == 0 {
				return nil
			}
			lms := make([]*int64, n)
			for i := 0; i < n; i++ {
				lms[i] = &lastModifieds[i]
			}
			return qHandler.AfterOtherQueues(lms, Default_WaitDuration)
		}
	)

	for i, config := range configs {
		cs[i], err = sq.NewConsumer(
			config,
			qHandler.NewDelayHandler(
				newHandler(&counts[i], &lastModifieds[i], handlers[i]),
				newWaitFunc(i),
			),
			newMessage,
		)
		if err != nil {
			return nil, err
		}
	}

	return &MultiConsumer{
		cs:     cs,
		counts: counts,
	}, nil
}

func (mc *MultiConsumer) StopAndWait() {
	for _, c := range mc.cs {
		c.StopAndWait()
	}
}

func (mc *MultiConsumer) State() []int64 {
	return mc.counts
}

func (mc *MultiConsumer) GetConsumer(index int) *sq.Consumer {
	if index >= len(mc.cs) {
		return nil
	}
	return mc.cs[index]
}
