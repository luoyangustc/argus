package job

import (
	"context"

	"qiniu.com/argus/censor_private/proto"
)

var _ IJob = (*MonitorActive)(nil)

type MonitorActive struct {
	monitor    IJob
	processJob IJob
}

func NewMonitorActive(
	ctx context.Context, setId string, url string, interval int,
	mimeTypes []proto.MimeType, worker IWorker, init bool, d *Dispatcher,
) IJob {
	return &MonitorActive{
		monitor:    NewMonitor(ctx, setId, url, interval, mimeTypes, d),
		processJob: NewProcessJob(ctx, setId, mimeTypes, worker, init),
	}
}

func (m *MonitorActive) Run() error {
	if err := m.monitor.Run(); err != nil {
		return err
	}
	if err := m.processJob.Run(); err != nil {
		return err
	}
	return nil
}

func (m *MonitorActive) Stop() error {
	if err := m.monitor.Stop(); err != nil {
		return err
	}
	if err := m.processJob.Stop(); err != nil {
		return err
	}
	return nil
}

func (m *MonitorActive) Notify() {
	m.monitor.Notify()
	m.processJob.Notify()
}
