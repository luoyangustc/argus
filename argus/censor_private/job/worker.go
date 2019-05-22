package job

import (
	"context"

	xlog "github.com/qiniu/xlog.v1"
	utility "qiniu.com/argus/argus/com/util"
	"qiniu.com/argus/censor_private/proto"
	"qiniu.com/argus/censor_private/util"
)

type IWorker interface {
	Size() int
	Pool() chan *WorkerJob
}

type WorkerConfig struct {
	WorkerSize         int
	ImageServiceConfig *proto.OuterServiceConfig
	VideoServiceConfig *proto.OuterServiceConfig
}

type worker struct {
	ctx          context.Context
	size         int
	ch           chan *WorkerJob
	imageService *CensorImageService
	videoService *CensorVideoService
}

type WorkerJob struct {
	ctx              context.Context
	uri              string
	cutIntervalMsecs int
	mimeType         proto.MimeType
	scenes           []proto.Scene
	f                WorkerJobFunc
}

type WorkerJobFunc func(context.Context, interface{}, *util.ErrorInfo)

func NewWorker(ctx context.Context, config *WorkerConfig) IWorker {
	w := &worker{
		ctx:          ctx,
		size:         config.WorkerSize,
		ch:           make(chan *WorkerJob, config.WorkerSize),
		imageService: NewCensorImageService(config.ImageServiceConfig),
		videoService: NewCensorVideoService(config.VideoServiceConfig),
	}

	for i := 0; i < w.size; i++ {
		go func() {
			for job := range w.ch {
				_ = w.processJob(utility.SpawnContext(job.ctx), job)
			}
		}()
	}
	return w
}

func (w *worker) Size() int {
	return w.size
}

func (w *worker) Pool() chan *WorkerJob {
	return w.ch
}

func (w *worker) processJob(ctx context.Context, job *WorkerJob) error {
	xl := xlog.FromContextSafe(ctx)
	_ = xl

	var (
		ret interface{}
		err *util.ErrorInfo
	)

	switch job.mimeType {
	case proto.MimeTypeImage:
		ret, err = w.imageService.Censor(ctx, job.uri, job.scenes)
	case proto.MimeTypeVideo:
		ret, err = w.videoService.Censor(ctx, job.uri, job.cutIntervalMsecs, job.scenes)
	case proto.MimeTypeUnknown:
		// TODO detect file type with file content
		// AND then update entry.MimeType

		//TODO if not in mimetypes, skip or maybe remove from db
		return nil
	default:
		return nil
	}

	if job.f != nil {
		job.f(ctx, ret, err)
	}
	return nil
}
