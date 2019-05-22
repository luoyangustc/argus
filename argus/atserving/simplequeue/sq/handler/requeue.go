package handler

import (
	"context"

	"qiniu.com/argus/atserving/simplequeue/sq"
)

var _ sq.Handler = RequeueHandler{}

type RequeueHandler struct {
	h           sq.Handler
	needRequeue func(context.Context, []byte) (bool, []byte, string, *sq.Producer)
}

func NewRequeueHandler(
	h sq.Handler,
	needRequeue func(context.Context, []byte) (bool, []byte, string, *sq.Producer),
) RequeueHandler {
	return RequeueHandler{
		h:           h,
		needRequeue: needRequeue,
	}
}

func (h RequeueHandler) HandleMessage(ctx context.Context, m sq.Message) error {

	if h.needRequeue != nil {
		if ok, newMessage, topic, p := h.needRequeue(ctx, m.GetBody()); ok {
			if err := p.Publish(topic, newMessage); err != nil {
				return err
			}
			return nil
		}
	}

	return h.h.HandleMessage(ctx, m)
}

//----------------------------------------------------------------------------//

func NeedRequeue(
	check func([]byte) bool,
	topic string,
	p *sq.Producer,
) func(context.Context, []byte) (bool, []byte, string, *sq.Producer) {

	return func(
		ctx context.Context,
		message []byte,
	) (bool, []byte, string, *sq.Producer) {

		return check(message), message, topic, p

	}

}
