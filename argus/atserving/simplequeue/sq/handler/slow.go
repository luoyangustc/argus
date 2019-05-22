package handler

import (
	"context"
	"time"

	"qiniu.com/argus/atserving/simplequeue/sq"
)

var _ sq.Handler = SlowHandler{}

type SlowHandler struct {
	h            sq.Handler
	tickDuration time.Duration
	timeout      time.Duration
}

func NewSlowHandler(
	h sq.Handler,
	tickDuration time.Duration,
	timeout time.Duration,
) SlowHandler {
	return SlowHandler{
		h:            h,
		tickDuration: tickDuration,
		timeout:      timeout,
	}
}

func (h SlowHandler) HandleMessage(ctx context.Context, m sq.Message) error {

	var (
		ticker                        = time.NewTicker(h.tickDuration)
		cancelFunc context.CancelFunc = nil
		c          chan error         = make(chan error)
	)

	defer ticker.Stop()

	if h.timeout > 0 {
		ctx, cancelFunc = context.WithTimeout(ctx, h.timeout)
	} else {
		ctx, cancelFunc = context.WithCancel(ctx)
	}
	defer cancelFunc()

	go func() {
		var err error
		defer func() { c <- err }()
		err = h.h.HandleMessage(ctx, m)
	}()

	for {
		select {
		case err := <-c:
			return err
		case <-ticker.C:
			m.Touch()
		}
	}

	return nil

}
