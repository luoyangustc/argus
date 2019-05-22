package job

import (
	"context"

	xlog "github.com/qiniu/xlog.v1"
	"qiniu.com/argus/censor_private/dao"
	"qiniu.com/argus/censor_private/proto"
)

const (
	FETCH_SIZE = 1000
)

var _ IJob = (*ProcessTask)(nil)

// 用于处理固定数量的任务，当处理到无新数据时即结束
// 收到停止信号时，会立刻停止运行
// 任务运行完后会设置set的状态为已完成
type ProcessTask struct {
	ctx        context.Context
	cancel     context.CancelFunc
	setId      string
	mimeTypes  []proto.MimeType
	worker     IWorker
	dispatcher *Dispatcher
}

func NewProcessTask(
	ctx context.Context, setId string, mimeTypes []proto.MimeType,
	worker IWorker, d *Dispatcher,
) IJob {
	cancelCtx, cancel := context.WithCancel(ctx)
	return &ProcessTask{
		ctx:        cancelCtx,
		cancel:     cancel,
		setId:      setId,
		mimeTypes:  mimeTypes,
		worker:     worker,
		dispatcher: d,
	}
}

func (p *ProcessTask) Run() error {
	go func() {
		var (
			xl     = xlog.FromContextSafe(p.ctx)
			marker string
		)

		for {
			// 一次获取1000条
			entries, err := dao.EntryDao.FindPending(p.setId, marker, "", FETCH_SIZE)
			if err != nil {
				xl.Errorf("get entry failed: %v", err)
				_ = p.dispatcher.Stop(p.ctx, p.setId)
				return
			}

			// 无新数据，处理完毕
			if len(entries) == 0 {
				// TODO 所有job在worker中完成了才算complete
				// TODO 通过回调判断是否已全部处理完（总数与已完成数）
				_ = p.dispatcher.Complete(p.ctx, p.setId)
				return
			}

			// 更新补充点
			marker = entries[len(entries)-1].Id.Hex()

			for i := range entries {
				select {
				case <-p.ctx.Done():
					return
				default:
				}

				e := entries[i]
				set, err := dao.SetCache.MustGet(e.SetId)
				if err != nil {
					xl.Errorf("get set failed: %v", err)
					continue
				}

				job := &WorkerJob{
					ctx:              p.ctx,
					uri:              e.Uri,
					mimeType:         e.MimeType,
					cutIntervalMsecs: set.CutIntervalMsecs,
					scenes:           set.Scenes,
					f:                store(e.Id.Hex(), set.CutIntervalMsecs),
				}
				p.worker.Pool() <- job
			}
		}
	}()

	return nil
}

func (p *ProcessTask) Stop() error {
	p.cancel()
	return nil
}

func (p *ProcessTask) Notify() {}
