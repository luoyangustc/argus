package sq

import "github.com/nsqio/go-nsq"

type Producer struct {
	producer *nsq.Producer
}

func NewProducer(addr string) (*Producer, error) {

	p, err := nsq.NewProducer(addr, nsq.NewConfig())
	if err != nil {
		return nil, err
	}
	p.SetLogger(Logger, LogLevel)
	if err = p.Ping(); err != nil {
		return nil, err
	}
	return &Producer{producer: p}, nil
}

func (p *Producer) Publish(topic string, body []byte) error {
	return p.producer.Publish(topic, body)
}

func (p *Producer) Stop() { p.producer.Stop() }
