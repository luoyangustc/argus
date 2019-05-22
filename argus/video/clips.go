package video

import (
	"context"

	"github.com/qiniu/xlog.v1"

	"qiniu.com/argus/video/segment"
	"qiniu.com/argus/video/vframe"
)

type _ClipResult struct {
	ClipResult
	Labels []ResultLabel
}

type Clips interface {
	Next(context.Context) (_ClipResult, bool)
	Error() error
	Close()
}

func NewClips(job vframe.Job,
	evalFunc func(context.Context, []string) ([]interface{}, error),
	parseFunc func(context.Context, interface{}) ([]string, []float32, error),
) Clips {
	return &simpleClips{
		Job:       job,
		evalFunc:  evalFunc,
		parseFunc: parseFunc,
	}
}

//----------------------------------------------------------------------------//

var _ Clips = &simpleClips{}

type simpleClips struct {
	vframe.Job
	err error

	evalFunc  func(context.Context, []string) ([]interface{}, error)
	parseFunc func(context.Context, interface{}) ([]string, []float32, error)
}

func (cs *simpleClips) Next(ctx context.Context) (ret _ClipResult, ok bool) {
	var (
		xl = xlog.FromContextSafe(ctx)
	)
	var cr vframe.CutResponse
	cr, ok = <-cs.Job.Cuts()
	if !ok {
		return
	}
	ok = false

	var (
		_clip = cr.Result.Cut

		labels []string
		scores []float32
		err    error
	)
	{
		var eResults []interface{}
		eResults, err = cs.evalFunc(ctx, []string{_clip.URI})
		if err != nil {
			xl.Warnf("evals failed. %v", err)
			cs.err = err
			return
		}
		ret.ClipResult.Result = eResults[0]
		labels, scores, err = cs.parseFunc(ctx, ret.ClipResult.Result)
		if err != nil {
			xl.Warnf("parse eval result failed. %v", err)
			cs.err = err
			return
		}
	}

	for i, label := range labels {
		ret.Labels = append(ret.Labels, ResultLabel{Name: label, Score: scores[i]})
	}

	ok = true
	return
}

func (cs *simpleClips) Error() error {
	if cs.err != nil {
		return cs.err
	}
	return cs.Job.Error()
}
func (cs *simpleClips) Close() { cs.Job.Stop() }

//----------------------------------------------------------------------------//

type _ClipsVframeJob struct {
	Interval                    float64
	Skip, SkipCount, FrameGroup int
	last                        float64
	vframe.Job
}

func newClipsVframeJob(job vframe.Job, skip, fg int) *_ClipsVframeJob {
	_job, ok := job.(VframeJob)
	if ok {
		return &_ClipsVframeJob{Interval: _job.Params().Interval,
			Skip:       skip,
			FrameGroup: fg,
			last:       _job.Params().StartTime,
			Job:        job,
		}
	}

	return &_ClipsVframeJob{Job: job}
}

func (job *_ClipsVframeJob) Next(ctx context.Context) (cr vframe.CutResponse, ok bool) {

	for {
		cr, ok = <-job.Job.Cuts()
		if !ok {
			return
		}

		if job.SkipCount > 0 {
			job.SkipCount--
			continue
		}

		break
	}
	return
}

//----------------------------------------------------------------------------//

var _ Clips = &everyframeClips{}

type everyframeClips struct {
	*_ClipsVframeJob
	offsets  []int64
	features [][]byte
	err      error

	evalFunc  func(context.Context, []string) ([]interface{}, error)
	parseFunc func(context.Context, interface{}) ([]string, []float32, error)
	topNFunc  func(context.Context, [][]byte) (interface{}, error)
}

func NewEveryFrameClips(job vframe.Job, skip, fg int,
	evalFunc func(context.Context, []string) ([]interface{}, error),
	parseFunc func(context.Context, interface{}) ([]string, []float32, error),
	topNFunc func(context.Context, [][]byte) (interface{}, error),
) *everyframeClips {
	return &everyframeClips{
		_ClipsVframeJob: newClipsVframeJob(job, skip, fg),
		offsets:         make([]int64, 0),
		features:        make([][]byte, 0),
		evalFunc:        evalFunc,
		parseFunc:       parseFunc,
		topNFunc:        topNFunc,
	}
}

func (ec *everyframeClips) Next(ctx context.Context) (ret _ClipResult, ok bool) {
	var (
		xl = xlog.FromContextSafe(ctx)
	)

	for len(ec.features) < ec._ClipsVframeJob.FrameGroup {
		var cr vframe.CutResponse
		cr, ok = ec._ClipsVframeJob.Next(ctx)
		if !ok {
			return
		}

		results, err := ec.evalFunc(ctx, []string{cr.Result.Cut.URI})
		if err != nil {
			xl.Warnf("evals failed. %v", err)
			ec.err = err
			return
		}

		ec.features = append(ec.features, (results[0]).([]byte))
		ec.offsets = append(ec.offsets, cr.Result.Cut.Offset)
	}

	eResult, err := ec.topNFunc(ctx, ec.features)
	if err != nil {
		xl.Warnf("topNFunc failed. %v", err)
		ec.err = err
		return
	}

	ret.ClipResult.Result = (eResult).([]map[string]float32)
	ret.ClipResult.OffsetBegin = ec.offsets[0]
	ret.ClipResult.OffsetEnd = ec.offsets[len(ec.offsets)-1]

	labels, scores, err := ec.parseFunc(ctx, ret.ClipResult.Result)
	if err != nil {
		xl.Warnf("parse eval result failed. %v", err)
		ec.err = err
		return
	}

	for i, label := range labels {
		ret.Labels = append(ret.Labels, ResultLabel{Name: label, Score: scores[i]})
	}

	if ec._ClipsVframeJob.Skip < ec._ClipsVframeJob.FrameGroup {
		ec.features = ec.features[ec._ClipsVframeJob.Skip:]
		ec.offsets = ec.offsets[ec._ClipsVframeJob.Skip:]
	} else {
		ec.features = ec.features[:0:0]
		ec.offsets = ec.offsets[:0:0]
	}

	ec._ClipsVframeJob.SkipCount = ec._ClipsVframeJob.Skip - ec._ClipsVframeJob.FrameGroup

	ok = true
	return
}

func (ec *everyframeClips) Error() error {
	if ec.err != nil {
		return ec.err
	}
	return ec.Job.Error()
}
func (ec *everyframeClips) Close() { ec.Job.Stop() }

////////////////////////////////////////////////////////////////////////////////

type SegmentJob interface {
	segment.Job
	Params() segment.SegmentParams
	OriginParams() segment.SegmentParams
}

type _SegmentJob struct {
	segment.Job
	actual, origin segment.SegmentParams
}

func (job *_SegmentJob) Clips() <-chan segment.ClipResponse  { return job.Job.Clips() }
func (job *_SegmentJob) Error() error                        { return job.Job.Error() }
func (job *_SegmentJob) Stop()                               { job.Job.Stop() }
func (job *_SegmentJob) Params() segment.SegmentParams       { return job.actual }
func (job *_SegmentJob) OriginParams() segment.SegmentParams { return job.origin }

//----------------------------------------------------------------------------//

var _ segment.Job = &segmentJob{}

type segmentJob struct {
	segment.Job

	openChs []bool
	chs     []chan segment.ClipResponse

	ctx    context.Context
	cancel context.CancelFunc
	err    error
}

func newSegmentJob(ctx context.Context, sfJob segment.Job) *segmentJob {
	ctx, cancel := context.WithCancel(ctx)
	job := &segmentJob{
		Job:     sfJob,
		openChs: []bool{true},
		chs: []chan segment.ClipResponse{
			make(chan segment.ClipResponse),
		},
		ctx:    ctx,
		cancel: cancel,
	}
	// go job.run()
	return job
}

func (job *segmentJob) Clips() <-chan segment.ClipResponse { return job.chs[0] }
func (job *segmentJob) Error() error                       { return job.err }
func (job *segmentJob) Stop()                              { job.cancel() }

func (job *segmentJob) run() {

	var (
		xl = xlog.FromContextSafe(job.ctx)
	)

	for {
		var end = false
		select {
		case clip, ok := <-job.Job.Clips():
			if !ok {
				end = true
				break
			}
			for i, ch := range job.chs {
				if job.openChs[i] {
					ch <- clip
				}
			}
		case <-job.ctx.Done():
			xl.Infof("cancel segment job")
			job.Job.Stop()
			end = true
			break
		}
		if end {
			break
		}
	}

	for _, ch := range job.chs {
		close(ch)
	}

	if err := job.Job.Error(); err != nil {
		job.err = err
	}
}

//----------------------------------------------------------------------------//

func spawnSegmentJob(ctx context.Context, _job segment.Job, n int) []segment.Job {
	_sjob, _ok := _job.(SegmentJob)
	job := newSegmentJob(ctx, _job)
	jobs := make([]segment.Job, 0, n)
	if _ok {
		jobs = append(jobs, &_SegmentJob{Job: job, actual: _sjob.Params(), origin: _sjob.OriginParams()})
	} else {
		jobs = append(jobs, job)
	}
	for i := 1; i < n; i++ {
		job.chs = append(job.chs, make(chan segment.ClipResponse))
		job.openChs = append(job.openChs, true)
		if _ok {
			jobs = append(jobs,
				&_SegmentJob{
					Job:    newSpawnSegmentJob(ctx, job, len(job.chs)-1),
					actual: _sjob.Params(),
					origin: _sjob.OriginParams(),
				})
		} else {
			jobs = append(jobs, newSpawnSegmentJob(ctx, job, len(job.chs)-1))
		}
	}
	go job.run()
	return jobs
}

type _SpawnSegmentJob struct {
	*segmentJob
	index int
}

func newSpawnSegmentJob(
	ctx context.Context, job *segmentJob, index int,
) *_SpawnSegmentJob {
	return &_SpawnSegmentJob{
		segmentJob: job,
		index:      index,
	}
}

func (job *_SpawnSegmentJob) Clips() <-chan segment.ClipResponse { return job.chs[job.index] }
func (job *_SpawnSegmentJob) Error() error                       { return job.segmentJob.Error() }
func (job *_SpawnSegmentJob) Stop() {
	job.segmentJob.openChs[job.index] = false
}
