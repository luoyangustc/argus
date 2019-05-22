package job

import (
	"context"
	"errors"
	"sync/atomic"
	"time"

	"github.com/qiniu/xlog.v1"

	. "qiniu.com/argus/bjob/proto"
)

type MasterNodeConfig struct {
	MaxPool          int32             `json:"pool_max"`
	ConcurrentPreJob int32             `json:"concurrent_pre_job"`
	TopicDefault     string            `json:"topic_default"`
	TopicOptions     map[string]string `json:"topic_options"`
}

type MasterNode struct {
	MasterNodeConfig
	JobCreator

	MQ
	*AsyncC

	ch            chan bool
	count_running int32
}

func NewMasterNode(
	config MasterNodeConfig,
	mq MQ, client *AsyncC,
	creator JobCreator,
) *MasterNode {
	return &MasterNode{
		MasterNodeConfig: config,
		JobCreator:       creator,
		MQ:               mq,
		AsyncC:           client,
		ch:               make(chan bool, config.MaxPool),
		count_running:    0,
	}
}

func (m *MasterNode) Run() {
	var (
		xl     = xlog.NewWith("MASTER")
		ctx    = xlog.NewContext(context.Background(), xl)
		ticker = time.NewTicker(time.Second * 10)
	)
	for {
		select {
		case <-ticker.C:
		case <-m.ch:
		}
		xl.Info("try execute jobs.")

		running := atomic.LoadInt32(&m.count_running)
		if running >= m.MaxPool {
			continue
		}

		jobs, _ := m.MQ.Execute(ctx, int(m.MaxPool-running))
		xl.Infof("execute jobs: %d", len(jobs))

		for _, job := range jobs {

			atomic.AddInt32(&m.count_running, 1)
			go func(ctx context.Context, job Job) {

				defer func() {
					atomic.AddInt32(&m.count_running, -1)
					m.ch <- true
				}()

				var (
					ch     = make(chan bool)
					ticker = time.NewTicker(time.Second * 10) // TODO config
				)

				go func(ctx context.Context) {
					defer ticker.Stop()
					for {
						var done = false
						select {
						case <-ticker.C:
							_ = m.MQ.Touch(ctx, &job)
						case <-ch:
							done = true
							break
						}
						if done {
							break
						}
					}
				}(ctx)

				defer func() {
					close(ch)
				}()

				var err error
				xl.Infof("runOneJob, %s", job.ID)
				job.Result, err = m.runOneJob(ctx, job)
				if err != nil {
					job.Error = err.Error()
				}
				m.hookOneJob(ctx, &job)
				_ = m.MQ.Finish(ctx, job)

			}(context.Background(), job)
		}
	}
}

func (m *MasterNode) hookOneJob(ctx context.Context, job *Job) {
	xl := xlog.FromContextSafe(ctx)
	xl.Infof("HOOK: %s", job.HookURL)
	if job.HookURL == "" {
		return
	}

	var _err error = nil
	if job.Error != "" {
		_err = errors.New(job.Error)
	}
	err := JobHook(ctx, job.HookURL, job.Result, _err,
		time.Second*10, time.Minute*5)
	if err != nil {
		job.HookError = err.Error()
	}
}

func (m *MasterNode) runOneJob(ctx context.Context, job Job) ([]byte, error) {

	var (
		xl = xlog.FromContextSafe(ctx)
	)
	job.Env.JID = job.ID
	master, err := m.JobCreator.NewMaster(ctx, job.Request, job.Env)
	if err != nil {
		xl.Warnf("new master failed. %v", err)
		return nil, err
	}

	var concurrent = m.ConcurrentPreJob
	if concurrent == 0 {
		concurrent = 100
	}
	ch := make(chan byte, concurrent)
	for i := 0; i < int(concurrent); i++ {
		ch <- 0x01
	}

	var count int32 = 0

	for {
		<-ch

		taskBody, kind, ok := master.NextTask(ctx)
		if !ok {
			break
		}

		atomic.AddInt32(&count, 1)
		_ = m.AsyncC.Publish(ctx, kind, taskBody,
			func(ctx context.Context, resultBody []byte, err error) {
				defer func() {
					atomic.AddInt32(&count, -1)
					ch <- 0x01
				}()

				_ = master.AppendResult(ctx,
					&_TaskResult{
						_Task:   &_Task{_Value: taskBody},
						_Result: resultBody,
						_Error:  err,
					},
				)
			},
		)
	}

	ticker := time.NewTicker(time.Second * 5)
	t1 := time.Now()
	last := atomic.LoadInt32(&count)
	for now := range ticker.C {
		if atomic.LoadInt32(&count) <= 0 {
			ticker.Stop()
			break
		}
		if curr := atomic.LoadInt32(&count); curr != last {
			last = curr
			t1 = time.Now()
			continue
		}
		if now.Sub(t1) > time.Hour+time.Minute*5 { // 大于worker单位超时
			ticker.Stop()
			break
		}
	}
	return master.Result(ctx)
}
