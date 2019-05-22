package job

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"time"

	xlog "github.com/qiniu/xlog.v1"
	"gopkg.in/mgo.v2/bson"
	"qiniu.com/argus/censor_private/dao"
	"qiniu.com/argus/censor_private/proto"
	"qiniu.com/argus/censor_private/util"
)

var _ IJob = (*ProcessJob)(nil)

// 用于处理不固定数量的任务，未停止前将一直轮循处理新数据
// 收到停止信号时，会继续运行直到处理完此刻前已入库的所有数据
type ProcessJob struct {
	ctx       context.Context
	cancel    context.CancelFunc
	setId     string
	mimeTypes []proto.MimeType
	worker    IWorker
	stop      bool   // 任务停止信号
	endMarker string // 当任务收到结束信号时，任务将继续运行到该marker为止
	init      bool   // 是否为初始状态
	notify    chan struct{}
	sync.Mutex
}

func NewProcessJob(
	ctx context.Context, setId string, mimeTypes []proto.MimeType,
	worker IWorker, init bool,
) IJob {
	cancelCtx, cancel := context.WithCancel(ctx)
	return &ProcessJob{
		ctx:       cancelCtx,
		cancel:    cancel,
		setId:     setId,
		mimeTypes: mimeTypes,
		worker:    worker,
		init:      init,
		notify:    make(chan struct{}),
	}
}

func (p *ProcessJob) Run() error {
	go func() {
		var (
			xl          = xlog.FromContextSafe(p.ctx)
			ticker      = time.NewTicker(15 * time.Second)
			supplyCh    = make(chan struct{})
			marker      string // 本次任务的entry起点
			countRemain int32
		)

		if !p.init {
			// 不为初始状态时，从当前Id最大的entry开始处理
			// 若为初始状态，则marker初始为空，以处理之前服务停止时可能遗留下的未处理的entry
			marker = getMaxEntryId(p.setId)
		}

		for {
			// 等待计时timeout或补充队列信号
			select {
			case <-p.ctx.Done():
				return
			case <-p.notify:
			case <-supplyCh:
			case <-ticker.C:
			}

			// 若当前待处理队列已足够，不操作
			remain := atomic.LoadInt32(&countRemain)
			if int(remain) >= 2*p.worker.Size() {
				continue
			}

			// 获取任务停止的相关信息
			p.Lock()
			endMarker := p.endMarker
			stop := p.stop
			p.Unlock()

			// 从上次的补充点开始补充待处理队列
			entries, err := dao.EntryDao.FindPending(p.setId, marker, endMarker, 2*p.worker.Size()-int(remain))
			if err != nil {
				xl.Errorf("get entry failed: %v", err)
			}
			if len(entries) == 0 {
				if stop {
					// 已收到停止信号，结束任务
					return
				} else {
					// 未收到停止信号，继续轮循获取entry
					continue
				}
			}

			// 更新补充点及队列count
			marker = entries[len(entries)-1].Id.Hex()
			atomic.AddInt32(&countRemain, int32(len(entries)))

			go func() {
				for i := range entries {
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

					// 更新队列count
					atomic.AddInt32(&countRemain, -1)
				}

				go func() {
					// 发送补充信号，及时补充
					supplyCh <- struct{}{}
				}()
			}()
		}
	}()

	return nil
}

func (p *ProcessJob) Stop() error {
	p.Lock()
	defer p.Unlock()
	// 设置结束点，job继续执行到checkPoint为止
	p.endMarker = getMaxEntryId(p.setId)
	p.stop = true
	return nil
}

func (p *ProcessJob) Notify() {
	go func() {
		p.notify <- struct{}{}
	}()
}

func store(entryId string, cutInterval int) WorkerJobFunc {
	return func(ctx context.Context, ret interface{}, retErr *util.ErrorInfo) {
		go func() {
			if retErr != nil &&
				retErr.Error() == context.Canceled.Error() {
				// 过滤cancel context造成的error
				return
			}

			var err error
			switch _ret := ret.(type) {
			case *proto.ImageCensorResult:
				err = patchImageResult(entryId, _ret, retErr)
			case *proto.VideoCensorResult:
				err = patchVideoResult(entryId, cutInterval, _ret, retErr)
			default:
			}

			if err != nil {
				xl := xlog.FromContextSafe(ctx)
				xl.Errorf("patch result error: %v", entryId, err)
			}
		}()
	}
}

func getMaxEntryId(setId string) string {
	var id string
	entries, _, _ := dao.EntryDao.Query(&dao.EntryFilter{SetId: setId}, "", 1)

	if len(entries) > 0 {
		id = entries[0].Id.Hex()
	}
	return id
}

func patchImageResult(entryId string, ret *proto.ImageCensorResult, retErr *util.ErrorInfo) error {
	err := dao.EntryDao.Patch(entryId,
		bson.M{
			"original":   ret,
			"error":      retErr,
			"created_at": time.Now().Unix(),
		})
	return err
}

func patchVideoResult(entryId string, cutInterval int, ret *proto.VideoCensorResult, retErr *util.ErrorInfo) error {
	var (
		entryRet = proto.CommonResult{
			Suggestion: ret.Suggestion,
			Scenes:     make(map[proto.Scene]proto.CommonSuggestion),
		}
		cutNum   int
		coverUri string
	)

	for s, v := range ret.Scenes {
		// entry中只存储video的审核结果，不存储帧结果
		entryRet.Scenes[s] = proto.CommonSuggestion{Suggestion: v.Suggestion}

		// 获取帧长度及封面uri
		if cutNum == 0 {
			cutNum = len(v.Cuts)
			if len(v.Cuts) > 0 {
				coverUri = v.Cuts[0].Uri
			}
		}

		// 检验帧长度
		if cutNum != len(v.Cuts) {
			return errors.New("invalid video result, cut numbers of scenes are different")
		}

	}
	if cutNum <= 0 {
		return errors.New("invalid video result, do not have any cut")
	}

	err := dao.EntryDao.Patch(entryId,
		bson.M{
			"original":           entryRet,
			"error":              retErr,
			"cover_uri":          coverUri,
			"cut_interval_msecs": cutInterval,
			"created_at":         time.Now().Unix(),
		})
	if err != nil {
		return err
	}

	// 存储帧结果
	videoCut := make([]*proto.VideoCut, cutNum)
	for scene, val := range ret.Scenes {
		for i, cut := range val.Cuts {
			if videoCut[i] == nil {
				videoCut[i] = &proto.VideoCut{
					EntryId: entryId,
					Uri:     cut.Uri,
					Offset:  cut.Offset,
					Original: &proto.OriginalSuggestion{
						Suggestion: proto.SuggestionPass,
						Scenes:     make(map[proto.Scene]interface{}),
					},
				}
			}

			videoCut[i].Original.Scenes[scene] = proto.ImageSceneResult{
				Suggestion: cut.Suggestion,
				Details:    cut.Details,
			}
			videoCut[i].Original.Suggestion = proto.MergeSuggestion(cut.Suggestion,
				videoCut[i].Original.Suggestion)
		}
	}

	// 只存储block和review的帧
	saveCut := make([]*proto.VideoCut, 0)
	for i := range videoCut {
		if videoCut[i].Original.Suggestion != proto.SuggestionPass {
			saveCut = append(saveCut, videoCut[i])
		}
	}
	err = dao.VideoCutDao.BatchInsert(saveCut)

	return err
}
