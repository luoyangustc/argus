package batch_entry_processor

import (
	"context"
	"time"

	"qiniu.com/argus/ccp/conf"
	"qiniu.com/argus/ccp/manual/client"
	"qiniu.com/argus/ccp/manual/dao"
	"qiniu.com/argus/ccp/manual/enums"
	"qiniu.com/argus/ccp/manual/model"

	xlog "github.com/qiniu/xlog.v1"
)

type BatchEntryJobResult struct {
	conf.BatchEntryResultConf
	SetDao        dao.ISetDAO
	BatchEntryDao dao.IBatchEntryDAO
	CapClient     client.ICAPClient
	NotifyClient  client.ICcpNotify

	jobCh   chan *model.BatchEntryModel
	closeCh chan struct{}
}

func NewBatchEntryJobResult(
	config conf.BatchEntryResultConf,
	setDao *dao.ISetDAO,
	batchEntryDao *dao.IBatchEntryDAO,
	capClient *client.ICAPClient,
	notifyClient *client.ICcpNotify,
) *BatchEntryJobResult {
	return &BatchEntryJobResult{
		BatchEntryResultConf: config,
		SetDao:               *setDao,
		BatchEntryDao:        *batchEntryDao,
		CapClient:            *capClient,
		NotifyClient:         *notifyClient,
		closeCh:              make(chan struct{}),
		jobCh:                make(chan *model.BatchEntryModel, config.MaxPool),
	}
}

func (this *BatchEntryJobResult) Start(ctx context.Context) {
	var (
		xl = xlog.FromContextSafe(ctx)
	)
	xl.Info("batch entry job result is starting")
	defer xl.Info("batch entry job result started")
	go this.checkJobs(ctx)
	go this.handleJobs(ctx)
}

func (this *BatchEntryJobResult) Close() {
	close(this.closeCh)
}

func (this *BatchEntryJobResult) checkJobs(ctx context.Context) {
	ticker := time.NewTicker(time.Duration(this.BatchEntryResultConf.ChecktTime) * time.Second)
	for {
		select {
		case <-ticker.C:
			this.checkJobsProxy(ctx)
		case <-this.closeCh:
			return
		}
	}
}
func (this *BatchEntryJobResult) checkJobsProxy(ctx context.Context) {
	var (
		xl = xlog.FromContextSafe(ctx)
	)
	jobs, err := this.BatchEntryDao.QueryByStatus(ctx, enums.BatchEntryJobStatusEnd)
	if err != nil {
		xl.Errorf("this.BatchEntryDao.QueryByStatu error: %#v", err.Error())
	}

	for _, job := range jobs {
		this.jobCh <- model.FromBatchEntryInMgo(job)
	}
}

//==================================================================//
func (this *BatchEntryJobResult) handleJobs(ctx context.Context) {
	for i := 0; i < this.BatchEntryResultConf.MaxPool; i++ {
		go func() {
			for job := range this.jobCh {
				this.processJob(ctx, job)
			}
		}()
	}
}

func (this *BatchEntryJobResult) processJob(ctx context.Context, job *model.BatchEntryModel) {
	var (
		xl = xlog.FromContextSafe(ctx)
	)

	bFinish, err := this.CapClient.CheckJob(ctx, job.ImageSetID)
	if err != nil {
		//更新数据库的状态为fail
		err2 := this.BatchEntryDao.UpdateStatus(ctx, job.SetId, enums.BatchEntryJobStatusFailed)
		if err2 != nil {
			xl.Errorf("this.BatchEntryDao.UpdateStatus error: %v", err.Error())
			//TODO: 是否需要多次重试更新数据库？
		}
		//将失败结果发送给CCP-manager
		err = this.NotifyClient.CallBackCapResult(ctx, job.ImageSetID, err)
		if err != nil {
			xl.Errorf("this.NotifyClient.CallBackCapResult error: %v", err.Error())
			return
		}
	} else {
		if bFinish {
			//更新数据库状态为success
			err := this.BatchEntryDao.UpdateStatus(ctx, job.SetId, enums.BatchEntryJobStatusSuccess)
			if err != nil {
				xl.Errorf("this.BatchEntryDao.UpdateStatus: %v", err.Error())
				//TODO: 是否需要多次重试更新数据库？
			}
			//将成功结果发送给ccp-manager
			err = this.NotifyClient.CallBackCapResult(ctx, job.ImageSetID, nil)
			if err != nil {
				xl.Errorf("this.NotifyClient.CallBackCapResult error: %v", err.Error())
				return
			}
		}
	}
}
