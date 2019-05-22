package concerns

import (
	"context"
	"encoding/json"
	"strings"
	"time"

	httputil "github.com/qiniu/http/httputil.v1"
	xlog "github.com/qiniu/xlog.v1"
	"gopkg.in/mgo.v2/bson"
	"qiniu.com/argus/ccp/review/dao"
	"qiniu.com/argus/ccp/review/enums"
	"qiniu.com/argus/ccp/review/misc"
	"qiniu.com/argus/ccp/review/model"
)

type _BatchEntryJobProcessor struct {
	kClient *KodoClient

	worker  int
	jobCh   chan *model.BatchEntryJob
	closeCh chan struct{}

	logger *xlog.Logger
	ctx    context.Context
}

func NewBatchEntryJobProcessor(
	ctx context.Context,
	kClient *KodoClient,
	size int) *_BatchEntryJobProcessor {
	return &_BatchEntryJobProcessor{
		kClient: kClient,

		worker:  size,
		jobCh:   make(chan *model.BatchEntryJob, size),
		closeCh: make(chan struct{}),

		logger: xlog.FromContextSafe(ctx),
		ctx:    ctx,
	}
}

func (this *_BatchEntryJobProcessor) Start() {
	this.logger.Info("batch entry job processor is starting")
	defer this.logger.Info("batch entry job processor started")
	go this.subJobs()
	go this.pubJobs()
}

func (this _BatchEntryJobProcessor) Close() {
	close(this.jobCh)
	close(this.closeCh)
}

func (this *_BatchEntryJobProcessor) pubJobs() {
	ticker := time.NewTicker(15 * time.Second)
	for {
		select {
		case <-ticker.C:
			this.findNewJobs()
		case <-this.closeCh:
			return
		}
	}
}

func (this *_BatchEntryJobProcessor) subJobs() {
	for i := 0; i < this.worker; i++ {
		go func() {
			for job := range this.jobCh {
				this.processJob(job)
			}
		}()
	}
}

func (this *_BatchEntryJobProcessor) findNewJobs() {
	jobs, err := dao.BatchEntryJobDAO.Query(this.ctx, enums.BatchEntryJobStatusNew)
	if err != nil {
		this.logger.Errorf("dao.BatchEntryJobDAO.Query: %v", err)
		return
	}

	for _, job := range jobs {
		err := dao.BatchEntryJobDAO.StartJob(this.ctx, job.ID)
		if err == nil {
			this.jobCh <- job
		} else if err != dao.ErrNotFound {
			this.logger.Errorf("dao.BatchEntryJobDAO.StartJob: <%s>, %#v", job.ID.Hex(), err)
		}
	}
}

const (
	entryFlushBuffSize = 50
	normalEntryLimit   = 5000
)

func (this *_BatchEntryJobProcessor) processJob(job *model.BatchEntryJob) (err error) {
	this.logger.Infof("start processing job: <%s>", job.ID.Hex())
	defer func(t1 time.Time) {
		var code = 200
		if err != nil {
			code = httputil.DetectCode(err)
		}
		misc.RequestsCounter("processInventoryJob", code).Inc()
		misc.ResponseTime("processInventoryJob", code).Observe(float64(time.Since(t1) / time.Second))
	}(time.Now())

	defer func() {
		if r := recover(); r != nil {
			this.logger.Errorf("process job with panic error: %v", r)
		}
		this.logger.Infof("end process job: <%s>", job.ID.Hex())
	}()

	ak, sk, dms := this.kClient.GetBucketInfo(job.Uid, job.Bucket)
	lineIter := NewBucketKeyIter(*this.kClient.Kodo, ak, sk, job.Bucket, dms[0], job.Key)
	defer lineIter.Close()

	setDao, err := dao.EntrySetCache.GetDao(job.SetId)
	if err != nil {
		this.logger.Errorf("dao.EntrySetCache.GetDao: <%s> %v", job.SetId, err)
		_ = dao.BatchEntryJobDAO.UpdateStatus(this.ctx, job.ID, enums.BatchEntryJobStatusProcess, enums.BatchEntryJobStatusFailed)
		return
	}

	var (
		entries           []*model.Entry
		normalEntryisFull bool
		specialEntries    []*model.Entry
		line              string
		hasMore           bool
		lineNumber        int64
	)

	for {
		line, hasMore = lineIter.Next(this.ctx)

		if !hasMore {
			if !normalEntryisFull {
				_ = this.flushEntries(job.ID, setDao, entries, lineNumber)
			}

			_ = this.flushEntries(job.ID, setDao, specialEntries, lineNumber)

			_ = dao.BatchEntryJobDAO.UpdateLineNumber(this.ctx, job.ID, lineNumber)

			break
		}

		lineNumber++

		if len(line) == 0 {
			continue
		}

		fields := strings.Split(line, "\t")
		if len(fields) < 2 {
			continue
		}

		var entry model.Entry
		if err := json.Unmarshal([]byte(fields[1]), &entry); err != nil {
			this.logger.Errorf("json.Unmarshal line <%d> with error: %v", lineNumber, err)
			continue
		}

		// ignore entry without original
		if entry.Original == nil {
			this.logger.Warnf("miss entry original info: %#v", entry)
			continue
		}

		if err = entry.Patch(); err != nil {
			this.logger.Warnf("failed to Patching version from: %#v", entry.Version)
			continue
		}

		entry.SetId = job.SetId

		// check entry suggestion
		if entry.Original.Suggestion == enums.SuggestionPass {
			// only store the frist 5000 normal entries
			if normalEntryisFull {
				continue
			}
			entries = append(entries, &entry)
		} else {
			specialEntries = append(specialEntries, &entry)
		}

		// batch insert pass entries
		if len(entries) >= entryFlushBuffSize {
			normalEntryisFull = this.flushEntries(job.ID, setDao, entries, lineNumber)
			entries = make([]*model.Entry, 0)
		}

		// batch insert special entries
		if len(specialEntries) >= entryFlushBuffSize {
			_ = this.flushEntries(job.ID, setDao, specialEntries, lineNumber)
			specialEntries = make([]*model.Entry, 0)
		}
	}

	if err = dao.BatchEntryJobDAO.UpdateStatus(
		this.ctx,
		job.ID,
		enums.BatchEntryJobStatusProcess,
		enums.BatchEntryJobStatusSuccess,
	); err != nil {
		this.logger.Errorf(
			"dao.BatchEntryJobDAO.UpdateStatus(%d): %v",
			enums.BatchEntryJobStatusSuccess,
			err,
		)
	}
	return
}

func (this *_BatchEntryJobProcessor) flushEntries(jobId bson.ObjectId, setDao dao.EntryDAO, entries []*model.Entry, lineNumber int64) bool {
	if len(entries) == 0 {
		return false
	}

	// add recover logic
	defer func() {
		if r := recover(); r != nil {
			this.logger.Errorf("flushEntries with panic error: %v", r)
		}
	}()

	// check normal entry is more than the limit.
	if entries[0].Original.Suggestion == enums.SuggestionPass {
		setId := entries[0].SetId

		n, err := setDao.Count(this.ctx, setId, enums.SuggestionPass)
		if err != nil {
			this.logger.Errorf("setDao.Count: <%s> %v", setId, err)
		}

		if n >= normalEntryLimit {
			return true
		}
	}

	if err := setDao.BatchInsert(this.ctx, entries); err != nil {
		this.logger.Errorf("setDao.BatchInsert: <%s> %v", jobId.Hex(), err)
		return false
	}

	EntryCounter.CheckEntries(this.ctx, entries)
	_ = dao.BatchEntryJobDAO.UpdateLineNumber(this.ctx, jobId, lineNumber)

	return false
}
