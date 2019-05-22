package sq

import (
	"context"
	"errors"

	"github.com/nsqio/go-nsq"
)

//----------------------------------------------------------------------------//

type Message interface {
	GetBody() []byte
	Touch()

	Tag(MessageTag) string

	AsNSQMessage() (nsq.Message, bool) // 特定使用场景，比如监控等；非业务需要
}

//----------------------------------------------------------------------------//

var (
	ErrRequeue               error = errors.New("requeue")
	ErrRequeueWithoutBackoff error = errors.New("requeue without backoff")
)

//----------------------------------------------------------------------------//

type Handler interface {
	HandleMessage(context.Context, Message) error
}

type HandlerFunc func(context.Context, Message) error

func (h HandlerFunc) HandleMessage(ctx context.Context, m Message) error {
	return h(ctx, m)
}

//----------------------------------------------------------------------------//

type MessageTag interface{}

//----------------------------------------------------------------------------//
