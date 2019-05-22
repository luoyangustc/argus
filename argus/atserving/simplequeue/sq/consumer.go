package sq

import (
	"context"
	"time"

	"github.com/nsqio/go-nsq"
)

type ConsumerConfig struct {
	Addresses []string `json:"addresses" yaml:"addresses"`
	Topic     string   `json:"topic" yaml:"topic"`
	Channel   string   `json:"channel" yaml:"channel"`

	Concurrent  *int `json:"concurrent" yaml:"concurrent"`
	MaxInFlight *int `json:"max_in_flight" yaml:"max_in_flight"`

	MaxAttempts *uint16 `json:"max_attempts" yaml:"max_attempts"`

	MaxRequeueDelay     *time.Duration `json:"max_requeue_delay" yaml:"max_requeue_delay"`
	DefaultRequeueDelay *time.Duration `json:"default_requeue_delay" yaml:"default_requeue_delay"`

	MaxBackoffDuration *time.Duration `json:"max_backoff_duration" yaml:"max_backoff_duration"`
}

func NewConsumerConfig() ConsumerConfig {
	var (
		Concurrent  int = _DEFAULT_Concurrent
		MaxInFlight int = _DEFAULT_MaxInFlight

		MaxAttempts         uint16        = _DEFAULT_MaxAttempts
		MaxRequeueDelay     time.Duration = _DEFAULT_MaxRequeueDelay
		DefaultRequeueDelay time.Duration = _DEFAULT_DefaultRequeueDelay
		MaxBackoffDuration  time.Duration = _DEFAULT_MaxBackoffDuration
	)

	return ConsumerConfig{
		Concurrent:  &Concurrent,
		MaxInFlight: &MaxInFlight,

		MaxAttempts: &MaxAttempts,

		MaxRequeueDelay:     &MaxRequeueDelay,
		DefaultRequeueDelay: &DefaultRequeueDelay,

		MaxBackoffDuration: &MaxBackoffDuration,
	}
}

var (
	_DEFAULT_Concurrent  int = 1
	_DEFAULT_MaxInFlight int = 500

	_DEFAULT_MaxAttempts uint16 = 5

	_DEFAULT_MaxRequeueDelay     time.Duration = time.Second * 10
	_DEFAULT_DefaultRequeueDelay time.Duration = time.Second

	_DEFAULT_MaxBackoffDuration time.Duration = time.Second * 5
)

type Consumer struct {
	consumer *nsq.Consumer
}

func NewConsumer(
	config ConsumerConfig,
	handler Handler,
	newMessage func(Message) Message,
) (*Consumer, error) {

	_config := nsq.NewConfig()

	//_config.ClientID = "" // TODO
	//_config.Hostname = "" // TODO

	_config.LookupdPollInterval = time.Second * 5

	_config.MaxAttempts = *config.MaxAttempts
	_config.MaxInFlight = *config.MaxInFlight

	_config.MaxRequeueDelay = *config.MaxRequeueDelay
	_config.DefaultRequeueDelay = *config.DefaultRequeueDelay

	_config.MaxBackoffDuration = *config.MaxBackoffDuration

	_config.OutputBufferSize = 1024
	_config.OutputBufferTimeout = time.Millisecond * 5

	c, err := nsq.NewConsumer(config.Topic, config.Channel, _config)
	if err != nil {
		return nil, err
	}
	c.SetLogger(Logger, LogLevel)
	c.AddConcurrentHandlers(_Handler{h: handler}, *config.Concurrent)
	if err = c.ConnectToNSQLookupds(config.Addresses); err != nil {
		return nil, err
	}
	return &Consumer{
		consumer: c,
	}, nil
}

func (c *Consumer) StopAndWait() error {
	c.consumer.Stop()
	<-c.consumer.StopChan
	return nil
}

func (c *Consumer) ChangeMaxInFlight(maxInFlight int) {
	c.consumer.ChangeMaxInFlight(maxInFlight)
}

//----------------------------------------------------------------------------//

var _ nsq.Handler = _Handler{}

type _Handler struct {
	h          Handler
	newMessage func(Message) Message
}

func (h _Handler) HandleMessage(m *nsq.Message) error {

	m.DisableAutoResponse()

	go func() {

		var err error
		defer func() {
			if err == nil {
				m.Finish()
			} else {
				switch err {
				case ErrRequeueWithoutBackoff:
					m.RequeueWithoutBackoff(-1)
				case ErrRequeue:
					m.Requeue(-1)
				default:
					m.Requeue(-1)
				}
			}
		}()

		var _m Message = &_Message{m: m}
		if h.newMessage != nil {
			_m = h.newMessage(_m)
		}

		err = h.h.HandleMessage(context.Background(), _m)
	}()

	return nil

}

//----------------------------------------------------------------------------//

var _ Message = &_Message{}

type _Message struct {
	m *nsq.Message
}

func (m *_Message) GetBody() []byte           { return m.m.Body }
func (m *_Message) Touch()                    { m.m.Touch() }
func (m *_Message) Tag(tag MessageTag) string { return "" }

func (m *_Message) AsNSQMessage() (nsq.Message, bool) { return *m.m, true }
