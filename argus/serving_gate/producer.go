package gate

import (
	"context"

	"github.com/qiniu/xlog.v1"

	"qiniu.com/argus/atserving/model"
	"qiniu.com/argus/atserving/simplequeue/sq"
)

// Producer ...
type Producer interface {
	Publish(
		ctx context.Context,
		id string,
		cmd string, version *string,
		body []byte,
	) error
}

const (
	_TopicFirst = "first_"
)

var _ Producer = &producer{}

type producer struct {
	ps []*sq.Producer
}

// NewProducer ...
func NewProducer(ps []*sq.Producer) Producer {
	return &producer{
		ps: ps,
	}
}

func (p *producer) topic(ctx context.Context, cmd string, version *string) string {
	if version == nil {
		return _TopicFirst + cmd
	}
	return _TopicFirst + cmd + "_" + *version
}

func (p *producer) Publish(
	ctx context.Context,
	id string,
	cmd string, version *string,
	body []byte,
) error {
	var (
		xl    = xlog.FromContextSafe(ctx)
		topic = p.topic(ctx, cmd, version)
	)
	body = model.NewMessageBody(id, body)

	xl.Infof("publish. %s %s %d", id, topic, len(body))
	// TODO RETRY
	return p.ps[0].Publish(topic, body)
}
