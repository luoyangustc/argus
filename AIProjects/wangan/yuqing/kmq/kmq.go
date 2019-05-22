package kmq

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"qiniu.com/argus/AIProjects/wangan/yuqing"

	xlog "github.com/qiniu/xlog.v1"
	"qbox.us/api/kmqcli"
)

const (
	KMQ_QUEUE_NAME = "WAYQ"
	LIMIT          = 1000
)

type KMQ interface {
	Produce(context.Context, yuqing.Message, yuqing.SourceType, yuqing.MediaType) error
	Consume(context.Context, string, int) ([]yuqing.Job, string, error)
	Consume1(context.Context, string, int) ([]yuqing.Job, string, error)
}

type kmq struct {
	kmqcli.Config
	*kmqcli.Client
	Uid uint32
}

var _ KMQ = &kmq{}

func NewKMQ(conf kmqcli.Config, uid uint32) KMQ {
	k := &kmq{
		Config: conf,
		Client: kmqcli.New(&conf),
		Uid:    uid,
	}
	k.CreateQueue(uid, KMQ_QUEUE_NAME, 7, xlog.NewDummy())
	return k
}

func (k *kmq) Produce(ctx context.Context, msg yuqing.Message, source yuqing.SourceType, media yuqing.MediaType) error {
	job, _ := json.Marshal(yuqing.Job{
		URI:     msg.URI,
		Type:    media,
		Source:  source,
		Message: msg,
	})
	if code, err := k.Client.ProduceMessagesAdmin(k.Uid, KMQ_QUEUE_NAME, []string{string(job)}, xlog.FromContextSafe(ctx)); err != nil {
		return errors.New(fmt.Sprintf("produce kmq message %d, error: %s", code, err.Error()))
	}
	return nil
}

func (k *kmq) Consume(ctx context.Context, position string, limit int) ([]yuqing.Job, string, error) {
	var jobs []yuqing.Job
	code, messages, next, err := k.Client.ConsumeMessages(KMQ_QUEUE_NAME, position, limit, xlog.NewDummy())
	if err != nil {
		err = errors.New(fmt.Sprintf("consume kmq message %d, error: %s", code, err.Error()))
	}
	for _, message := range messages {
		var job yuqing.Job
		json.Unmarshal([]byte(message), &job)
		jobs = append(jobs, job)
	}
	return jobs, next, err
}

func (k *kmq) Consume1(ctx context.Context, position string, limit int) ([]yuqing.Job, string, error) {
	var (
		job1 struct {
			URI    string            `json:"uri"`
			Type   string            `json:"type"`
			Source yuqing.SourceType `json:"source"`
		}
		jobs []yuqing.Job
	)
	code, messages, next, err := k.Client.ConsumeMessages(KMQ_QUEUE_NAME, position, limit, xlog.NewDummy())
	if err != nil {
		err = errors.New(fmt.Sprintf("consume kmq message %d, error: %s", code, err.Error()))
	}
	for _, message := range messages {
		json.Unmarshal([]byte(message), &job1)
		job := yuqing.Job{
			URI:    job1.URI,
			Source: job1.Source,
		}
		switch job1.Type {
		case "image":
			job.Type = yuqing.MediaTypeImage
		case "video":
			job.Type = yuqing.MediaTypeVideo
		}
		jobs = append(jobs, job)
	}
	return jobs, next, err
}
