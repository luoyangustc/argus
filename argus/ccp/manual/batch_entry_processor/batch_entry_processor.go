package batch_entry_processor

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	xlog "github.com/qiniu/xlog.v1"
	"qiniu.com/argus/ccp/conf"
	"qiniu.com/argus/ccp/manual/client"
	"qiniu.com/argus/ccp/manual/dao"
	"qiniu.com/argus/ccp/manual/enums"
	"qiniu.com/argus/ccp/manual/model"
	"qiniu.com/argus/ccp/manual/saver"
	"qiniu.com/argus/censor/biz"
)

type BatchEntryJobProcessor struct {
	conf.BatchEntryProcessorConf
	kClient       *saver.KodoClient
	SetDao        dao.ISetDAO
	BatchEntryDao dao.IBatchEntryDAO
	CapClient     client.ICAPClient

	jobCh       chan *model.BatchEntryModel
	closeCh     chan struct{}
	bucketSaver saver.IBucketSaver
}

func NewBatchEntryJobProcessor(
	conf *conf.BatchEntryProcessorConf,
	kClient *saver.KodoClient,
	setDao *dao.ISetDAO,
	batchEntryDao *dao.IBatchEntryDAO,
	capClient *client.ICAPClient,
	bucketSaver *saver.IBucketSaver,
) *BatchEntryJobProcessor {
	return &BatchEntryJobProcessor{
		BatchEntryProcessorConf: *conf,
		kClient:                 kClient,
		SetDao:                  *setDao,
		BatchEntryDao:           *batchEntryDao,
		CapClient:               *capClient,
		jobCh:                   make(chan *model.BatchEntryModel, conf.MaxPool),
		closeCh:                 make(chan struct{}),
		bucketSaver:             *bucketSaver,
	}
}

func (this *BatchEntryJobProcessor) Start(ctx context.Context) {
	var (
		xl = xlog.FromContextSafe(ctx)
	)
	xl.Info("batch entry job processor is starting")
	defer xl.Info("batch entry job processor started")
	go this.subJobs(ctx)
	go this.pubJobs(ctx)
}

func (this *BatchEntryJobProcessor) Close() {
	close(this.jobCh)
	close(this.closeCh)
}

//==================================================================================
func (this *BatchEntryJobProcessor) pubJobs(ctx context.Context) {
	ticker := time.NewTicker(15 * time.Second)
	for {
		select {
		case <-ticker.C:
			this.findNewJobs(ctx)
		case <-this.closeCh:
			return
		}
	}
}

func (this *BatchEntryJobProcessor) findNewJobs(ctx context.Context) {
	var (
		xl = xlog.FromContextSafe(ctx)
	)
	jobs, err := this.BatchEntryDao.QueryByStatus(ctx, enums.BatchEntryJobStatusNew)
	if err != nil {
		xl.Errorf("dao.IBatchEntryJobDAO.QueryByStatus: %v", err)
		return
	}

	for _, job := range jobs {
		err := this.BatchEntryDao.StartJob(ctx, job.SetId)
		if err == nil {
			xl.Infof("find job: <%s>", job.SetId)
			this.jobCh <- model.FromBatchEntryInMgo(job)
		} else if err != ErrNotFound {
			xl.Errorf("dao.BatchEntryJobDAO.StartJob: <%s>, %#v", job.ID.Hex(), err)
		}
	}
}

//==================================================================================
func (this *BatchEntryJobProcessor) subJobs(ctx context.Context) {
	for i := 0; i < this.BatchEntryProcessorConf.MaxPool; i++ {
		go func() {
			for job := range this.jobCh {
				this.processJob(ctx, job)
			}
		}()
	}
}

func (this *BatchEntryJobProcessor) processJob(ctx context.Context, job *model.BatchEntryModel) {
	var (
		xl = xlog.FromContextSafe(ctx)
	)

	ak, sk, domain := this.kClient.GetBucketInfo(job.Uid, job.Bucket)

	taskIdLabel := time.Now().Format("20060102150304")
	lineNo := int64(0) //统计某个job中一共发给cap的task数

	for _, key := range job.Keys {
		xl.Infof("begin process job: %s, key: %s", job.SetId, key)
		iter := saver.NewBucketKeyIter(*this.kClient.Kodo, ak, sk, job.Bucket, domain, key, this.BatchEntryProcessorConf.Gzip)
		defer iter.Close()

		totalNo := int64(0) //统计某个job中某个文件一共被处理的行数
		this.beginCapJob(ctx, ak, sk, domain, key, &iter, job, taskIdLabel, &lineNo, &totalNo)
		xl.Infof("job %s , key %d total number is: %d", job.SetId, key, totalNo)
	}

	xl.Infof("job %s do cap number is: %d", job.SetId, lineNo)
	//结束任务 Image && Video
	if job.ImageSetID != "" {
		err := this.BatchEntryDao.UpdateStatus(ctx, job.SetId, enums.BatchEntryJobStatusEnd)
		if err != nil {
			xl.Errorf("this.BatchEntryDao.UpdateStatus err: %v", err.Error())
			return
		}

	}
	if job.VideoSetID != "" {
		err := this.BatchEntryDao.UpdateStatus(ctx, job.SetId, enums.BatchEntryJobStatusEnd)
		if err != nil {
			xl.Errorf("this.BatchEntryDao.UpdateStatus err: %v", err.Error())
			return
		}
	}
}

func (this *BatchEntryJobProcessor) beginCapJob(ctx context.Context,
	ak, sk, domain, key string, iter *saver.LineIter, job *model.BatchEntryModel, taskIdLabel string, lineno, totalno *int64) {

	var (
		xl          = xlog.FromContextSafe(ctx)
		noReviewBuf = bytes.NewBuffer(make([]byte, 0))
		bufLine     = 0

		imageTasks []*model.BatchTasksReq
		//videoTasks  []interface{}
	)

	var writer = func(ctx context.Context, buf *bytes.Buffer, bufLine *int, linestr string) {
		var (
			xl = xlog.FromContextSafe(ctx)
		)

		buf.WriteString(linestr)
		buf.WriteString("\n")
		(*bufLine)++

		if *bufLine >= this.BatchEntryProcessorConf.MaxFileLine {
			err := this.bucketSaver.SaveResult(ctx, job.SetId, string(enums.MimeTypeImage), buf, *bufLine)
			if err != nil {
				xl.Errorf("batch_entry_processor.bucketSaver error: %#v", err.Error())
			}
			buf.Reset()
			(*bufLine) = 0
		}
	}

	setInMgo, err := this.SetDao.QueryByID(ctx, job.SetId)
	if err != nil {
		xl.Errorf("batch_entry_processor.SetDao.QueryByID error: %#v", err.Error())
	}

	for {
		linestr, ok, err := (*iter).Next(ctx)

		if !ok || err != nil {
			xl.Errorf("iter.Next ok val: %#v", ok)
			if err != nil {
				xl.Errorf("iter.Next error: %#v", err.Error())
			}
			break
		}
		if linestr == "" {
			continue
		}

		result := model.BcpResultModel{}
		strs := strings.Split(linestr, "\t")
		if len(strs) < 1 {
			continue
		}

		(*totalno)++ //统计某个job中某个文件一共被处理的行数
		if len(strs) == 1 {
			//对于机审结果格式不对的记录，直接保存到不需要机审的文件中
			writer(ctx, noReviewBuf, &bufLine, linestr)
			continue
		}
		if len(strs) >= 2 {
			err = json.Unmarshal([]byte(strs[1]), &result)
			if err != nil {
				xl.Errorf("json.Unmarshal error: %#v", err.Error())
				//解析result错误，直接写回文件保存，不进人审
				writer(ctx, noReviewBuf, &bufLine, linestr)
				continue
			}
		}

		//图片
		if result.Mimetype == string(enums.MimeTypeImage) {
			if isImageCallCap(ctx, &result) {
				taskId := fmt.Sprintf("%s_%s_%d", job.ImageSetID, taskIdLabel, *lineno)
				(*lineno)++
				xl.Infof("call cap image lineno: %d", *lineno)
				var batchReq *model.BatchTasksReq
				if result.Error != "" {
					//TODO:前端增加处理逻辑，后端删除该端代码
					batchReq = model.AddInfoForErr(setInMgo.UID, taskId,
						strs[0], setInMgo.Image.Scenes)
				} else {
					batchReq = model.FromImageBcpResultModel(setInMgo.UID, taskId,
						strs[0], &result)
				}
				//xl.Infof("batchReq : %#v", batchReq)
				imageTasks = append(imageTasks, batchReq)

				if len(imageTasks) >= this.BatchEntryProcessorConf.MaxCapTask {
					xl.Infof("begin pushBatchTask of job: %s,  and the len is: %d", job.SetId, len(imageTasks))

					err := this.CapClient.PushBatchTask(ctx, job.ImageSetID, imageTasks)
					if err != nil {
						xl.Errorf("CAPClient.PushTasks err: %v", err)
						continue
					}
					imageTasks = []*model.BatchTasksReq{}
				}
			} else {
				// 不需要review的，直接写入结果文件
				writer(ctx, noReviewBuf, &bufLine, linestr)
			}
		}

		//视频
		if result.Mimetype == string(enums.MimeTypeVideo) {
			//TODO:需要review的视频
			if isVideoCallCap(ctx, &result) {

			} else {
				// 不需要review的，直接写入结果文件
				writer(ctx, noReviewBuf, &bufLine, linestr)
			}
		}
	}

	// send last images to cap
	if len(imageTasks) > 0 {
		xl.Infof("begin pushBatchTask last, and len is: %d", len(imageTasks))
		err := this.CapClient.PushBatchTask(ctx, job.ImageSetID, imageTasks)
		if err != nil {
			xl.Errorf("CAPClient.PushTasks err: %v", err)
			return
		}
	}

	if bufLine > 0 {
		err = this.bucketSaver.SaveResult(ctx, job.SetId, string(enums.MimeTypeImage), noReviewBuf, bufLine)
		if err != nil {
			xl.Errorf("batch_entry_processor.bucketSaver error: %#v", err.Error())
		}
		bufLine = 0
		noReviewBuf.Reset()
	}

	return
}

//==========================================================================
func isImageCallCap(ctx context.Context, req *model.BcpResultModel) bool {
	if req.Code != 200 || req.Error != "" {
		return true
	}
	var (
		xl             = xlog.FromContextSafe(ctx)
		censorResponse biz.CensorResponse
	)
	err := json.Unmarshal(req.Result, &censorResponse)
	if err != nil {
		xl.Errorf("json.Unmarshal error: %#v", err.Error())
		return true
	}

	if censorResponse.Suggestion == biz.PASS {
		return false
	}

	if censorResponse.Suggestion == "" {
		//对于机审结果为空的情况，在发送给labelx之前加一个默认值，类似这种情况{"code":200,"mimetype":"image","result":{"code":200}}
		req.Error = "bcp result is null"
	}
	return true
}

func isVideoCallCap(ctx context.Context, req *model.BcpResultModel) bool {
	return false
}

//==========================================================================
