package job

import (
	"context"
	"time"

	xlog "github.com/qiniu/xlog.v1"
	"qiniu.com/argus/censor_private/dao"
	"qiniu.com/argus/censor_private/proto"
	"qiniu.com/argus/censor_private/util"
)

var _ IJob = (*Monitor)(nil)

type Monitor struct {
	ctx        context.Context
	cancel     context.CancelFunc
	setId      string
	url        string
	interval   int
	mimeTypes  []proto.MimeType
	dispatcher *Dispatcher
}

func NewMonitor(ctx context.Context, setId string, url string,
	interval int, mimeTypes []proto.MimeType, d *Dispatcher,
) IJob {
	cancelCtx, cancel := context.WithCancel(ctx)
	return &Monitor{
		ctx:        cancelCtx,
		cancel:     cancel,
		setId:      setId,
		url:        url,
		interval:   interval,
		mimeTypes:  mimeTypes,
		dispatcher: d,
	}
}

func (m *Monitor) Run() error {
	go func() {
		xl := xlog.FromContextSafe(m.ctx)
		ticker := time.NewTicker(time.Duration(m.interval) * time.Second)
		reqTimeout := time.Duration(m.interval/2) * time.Second

		for {
			select {
			case <-m.ctx.Done():
				return
			case <-ticker.C:
			}

			// get url list
			bs, err := util.GetWithContent(m.ctx, reqTimeout, m.url)
			if err != nil {
				xl.Errorf("monitor get url content fail: %v", err)
				//TODO maybe write error in db
				continue
			}
			urls := util.ByteArrayToLines(bs)

			// insert in db
			count := dao.InsertEntries(m.ctx, m.setId, m.mimeTypes, urls)
			if count > 0 {
				m.dispatcher.Notify(m.setId)
				xl.Infof("set(%s) collect %d entries", m.setId, count)
			}
		}
	}()

	return nil
}

func (m *Monitor) Stop() error {
	m.cancel()
	return nil
}

func (m *Monitor) Notify() {}
