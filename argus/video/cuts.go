package video

import (
	"container/list"
	"container/ring"
	"context"

	"github.com/qiniu/xlog.v1"

	"qiniu.com/argus/video/vframe"
)

type CutResultWithLabels struct {
	CutResult
	Labels []ResultLabel
}

// func FOO() {
// 	var cuts Cuts
// 	for {
// 		ret, ok := cuts.Next(ctx)
// 		if !ok {
// 			break
// 		}
// 		// TODO
// 	}
// 	err = cuts.Erro()
// }
type Cuts interface {
	Next(context.Context) (CutResultWithLabels, bool)
	Error() error
	Close()
}

type CutsV2 interface {
	Append(context.Context, vframe.CutResponse) ([]CutResultWithLabels, bool, error)
	Close(context.Context) ([]CutResultWithLabels, error)
}

// func NewCuts(job vframe.Job,
// 	evalFunc func(context.Context, []string) ([]interface{}, error),
// 	parseFunc func(context.Context, interface{}) ([]string, []float32, error),
// ) Cuts {
// 	_job, ok := job.(VframeJob)
// 	if ok {
// 		return newSimpleCuts(
// 			NewSimpleCutsV2(_job.Params(), _job.OriginParams(), evalFunc, parseFunc),
// 			job)
// 	}
// 	return newSimpleCuts(
// 		&simpleCutsV2{_FixedTimeline: nil, evalFunc: evalFunc, parseFunc: parseFunc},
// 		job)
// }
func NewCuts(job vframe.Job, cutsV2 CutsV2) Cuts {
	return newSimpleCuts(cutsV2, job)
}

//----------------------------------------------------------------------------//

type simpleCutsV2 struct {
	*_FixedTimeline

	evalFunc  func(context.Context, []string) ([]interface{}, error)
	parseFunc func(context.Context, interface{}) ([]string, []float32, error)
}

func NewSimpleCutsV2(
	params, originParams *vframe.VframeParams,
	evalFunc func(context.Context, []string) ([]interface{}, error),
	parseFunc func(context.Context, interface{}) ([]string, []float32, error),
) CutsV2 {

	var tl *_FixedTimeline
	if params != nil && originParams != nil {
		if params.GetMode() == vframe.MODE_INTERVAL && params.Interval != originParams.Interval {
			tl = newFixedTimeline(
				int64(params.Interval*1000),
				int64(originParams.Interval*1000),
				[]int64{0},
			)
		}
	}

	return &simpleCutsV2{_FixedTimeline: tl, evalFunc: evalFunc, parseFunc: parseFunc}
}

func (cs *simpleCutsV2) Append(
	ctx context.Context, cr vframe.CutResponse,
) (rets []CutResultWithLabels, ok bool, err error) {
	var (
		xl = xlog.FromContextSafe(ctx)
	)

	if cs._FixedTimeline != nil {
		skip, _, isSelect := cs._FixedTimeline.Update(cr.Result.Cut.Offset)
		if skip {
			return
		}
		if isSelect&__FIXED_SELECT != __FIXED_SELECT {
			return
		}
	}

	var (
		_cut = cr.Result.Cut
		ret  CutResultWithLabels

		labels []string
		scores []float32
	)
	{
		var eResults []interface{}
		eResults, err = cs.evalFunc(ctx, []string{_cut.URI})
		if err != nil {
			xl.Warnf("evals failed. %v", err)
			return
		}
		ret.CutResult.Result = eResults[0]
		labels, scores, err = cs.parseFunc(ctx, ret.CutResult.Result)
		if err != nil {
			xl.Warnf("parse eval result failed. %v", err)
			return
		}
	}
	ret.CutResult.Offset = _cut.Offset
	ret.CutResult.URI = _cut.URI
	for i, label := range labels {
		ret.Labels = append(ret.Labels, ResultLabel{Name: label, Score: scores[i]})
	}

	ok = true
	rets = []CutResultWithLabels{ret}
	return
}

func (cs *simpleCutsV2) Close(ctx context.Context) ([]CutResultWithLabels, error) { return nil, nil }

var _ Cuts = &simpleCuts{}

type simpleCuts struct {
	CutsV2

	vframe.Job
	end bool
	err error

	cache *list.List
}

func newSimpleCuts(cutsV2 CutsV2, job vframe.Job) Cuts {
	return &simpleCuts{CutsV2: cutsV2, Job: job, cache: list.New()}
}

func (cs *simpleCuts) Next(ctx context.Context) (ret CutResultWithLabels, ok bool) {

	for {
		if cs.cache.Len() > 0 {
			ret = cs.cache.Remove(cs.cache.Front()).(CutResultWithLabels)
			return ret, true
		}
		if cs.end {
			return
		}

		var rets []CutResultWithLabels
		var err error

		var cr vframe.CutResponse
		cr, ok = <-cs.Job.Cuts()
		if !ok {
			cs.end = true
			rets, err = cs.CutsV2.Close(ctx)
		} else {
			rets, _, err = cs.Append(ctx, cr)
		}

		if err != nil {
			cs.err = err
			return
		}
		if len(rets) == 0 {
			continue
		}
		for i := 1; i < len(rets); i++ {
			cs.cache.PushBack(rets[i])
		}
		return rets[0], true
	}
}

func (cs *simpleCuts) Error() error {
	if cs.err != nil {
		return cs.err
	}
	return cs.Job.Error()
}
func (cs *simpleCuts) Close() { cs.Job.Stop() }

//----------------------------------------------------------------------------//

type fixedAggCutsV2 struct {
	*_FixedTimeline
	cache       *ring.Ring
	cacheSelect []int

	evalFunc  func(context.Context, []string, bool) ([]interface{}, error)
	aggFunc   func(context.Context, []interface{}) (interface{}, error)
	parseFunc func(context.Context, interface{}) ([]string, []float32, error)
}

func NewFixedAggCutsV2(
	interval, originInterval int64,
	fixedDurations []int64,
	evalFunc func(context.Context, []string, bool) ([]interface{}, error),
	aggFunc func(context.Context, []interface{}) (interface{}, error),
	parseFunc func(context.Context, interface{}) ([]string, []float32, error),
) CutsV2 {
	cuts := newFixedAggCutsV2(interval, originInterval, fixedDurations)
	cuts.evalFunc = evalFunc
	cuts.aggFunc = aggFunc
	cuts.parseFunc = parseFunc
	return cuts
}

func newFixedAggCutsV2(
	interval, originInterval int64,
	fixedDurations []int64,
) *fixedAggCutsV2 {
	var (
		n   = len(fixedDurations)
		max = fixedDurations[n-1] - fixedDurations[0]

		selects = make([]int, n+1)
		last    = fixedDurations[n-1]
	)
	for i, d := range fixedDurations {
		selects[i+1] = 0 - int((last-d)/interval)
		if d == 0 {
			selects[0] = 0 - int((last-d)/interval)
		}
	}
	return &fixedAggCutsV2{
		_FixedTimeline: newFixedTimeline(interval, originInterval, fixedDurations),
		cache:          ring.New(int(max/interval) + 1),
		cacheSelect:    selects,
	}
}

func (cs *fixedAggCutsV2) update(
	ctx context.Context, cr *vframe.CutResponse,
) (rets []CutResultWithLabels, ok bool, err error) {

	type E struct {
		Offset  int64
		URI     string
		Select  byte
		EResult interface{}
	}

	var (
		xl       = xlog.FromContextSafe(ctx)
		mockCuts = list.New()

		eResult interface{}
		labels  []string
		scores  []float32
	)

	if cr == nil {
		for _, tail := range cs._FixedTimeline.Tail() {
			mockCuts.PushBack(E{Offset: tail.Offset, Select: tail.Select})
		}
	} else {

		skip, misses, isSelect := cs._FixedTimeline.Update(cr.Result.Cut.Offset)
		if skip {
			return
		}
		for _, miss := range misses {
			mockCuts.PushBack(E{Offset: miss.Offset, Select: miss.Select})
		}

		e := E{Offset: cr.Result.Cut.Offset, URI: cr.Result.Cut.URI, Select: isSelect}
		if isSelect&__FIXED_SELECT == __FIXED_SELECT {
			var eResults []interface{}
			if isSelect&__FIXED_TARGET == __FIXED_TARGET {
				eResults, err = cs.evalFunc(ctx, []string{cr.Result.Cut.URI}, false)
			} else {
				eResults, err = cs.evalFunc(ctx, []string{cr.Result.Cut.URI}, true)
			}
			if err != nil {
				xl.Warnf("evals failed. %v", err)
				return
			}

			e.EResult = eResults[0]
		}

		mockCuts.PushBack(e)
	}

	for {
		var _e = mockCuts.Front()
		if _e == nil {
			break
		}
		e := mockCuts.Remove(_e).(E)
		cs.cache.Value = CutResult{Offset: e.Offset, URI: e.URI, Result: e.EResult}
		cs.cache = cs.cache.Next()

		if e.Select&__FIXED_ROUND != __FIXED_ROUND {
			continue
		}
		if cs.cache.Move(cs.cacheSelect[0]-1).Value == nil { // 开始阶段
			continue
		}

		var notNil = false
		results := make([]interface{}, len(cs.cacheSelect)-1)
		for i, index := range cs.cacheSelect[1:] {
			r := cs.cache.Move(index - 1)
			if r.Value == nil {
				results[i] = nil
			} else {
				results[i] = r.Value.(CutResult).Result
				if results[i] != nil {
					notNil = true
				}
			}
		}
		if !notNil {
			continue
		}
		eResult, _ = cs.aggFunc(ctx, results)

		var ret CutResultWithLabels
		ret.CutResult.Result = eResult
		labels, scores, err = cs.parseFunc(ctx, ret.CutResult.Result)
		if err != nil {
			xl.Warnf("parse eval result failed. %v", err)
			return
		}
		ret.CutResult.Offset = cs.cache.Move(cs.cacheSelect[0] - 1).Value.(CutResult).Offset
		ret.CutResult.URI = cs.cache.Move(cs.cacheSelect[0] - 1).Value.(CutResult).URI
		for i, label := range labels {
			ret.Labels = append(ret.Labels, ResultLabel{Name: label, Score: scores[i]})
		}

		rets = append(rets, ret)
	}

	ok = cr != nil && len(rets) > 0 && rets[len(rets)-1].Offset == cr.Result.Cut.Offset
	return
}

func (cs *fixedAggCutsV2) Append(
	ctx context.Context, cr vframe.CutResponse,
) (rets []CutResultWithLabels, ok bool, err error) {
	return cs.update(ctx, &cr)
}

func (cs *fixedAggCutsV2) Close(ctx context.Context) (rets []CutResultWithLabels, err error) {
	rets, _, err = cs.update(ctx, nil)
	return
}

func NewFixedAggCuts(
	interval, originInterval int64,
	fixedDurations []int64,
	job vframe.Job,
	evalFunc func(context.Context, []string, bool) ([]interface{}, error),
	aggFunc func(context.Context, []interface{}) (interface{}, error),
	parseFunc func(context.Context, interface{}) ([]string, []float32, error),
) Cuts {
	cuts := &simpleCuts{CutsV2: NewFixedAggCutsV2(interval, originInterval, fixedDurations, evalFunc, aggFunc, parseFunc)}
	cuts.Job = job
	cuts.cache = list.New()
	return cuts
}

////////////////////////////////////////////////////////////////////////////////

type VframeJob interface {
	vframe.Job
	Params() vframe.VframeParams
	OriginParams() vframe.VframeParams
}

type _VframeJob struct {
	vframe.Job
	actual, origin vframe.VframeParams
}

func (job *_VframeJob) Cuts() <-chan vframe.CutResponse   { return job.Job.Cuts() }
func (job *_VframeJob) Error() error                      { return job.Job.Error() }
func (job *_VframeJob) Stop()                             { job.Job.Stop() }
func (job *_VframeJob) Params() vframe.VframeParams       { return job.actual }
func (job *_VframeJob) OriginParams() vframe.VframeParams { return job.origin }

//----------------------------------------------------------------------------//

var _ vframe.Job = &vframeJob{}

type vframeJob struct {
	vframe.Job

	openChs []bool
	chs     []chan vframe.CutResponse

	ctx    context.Context
	cancel context.CancelFunc
	err    error
}

func newVframeJob(ctx context.Context, vfJob vframe.Job) *vframeJob {
	ctx, cancel := context.WithCancel(ctx)
	job := &vframeJob{
		Job:     vfJob,
		openChs: []bool{true},
		chs: []chan vframe.CutResponse{
			make(chan vframe.CutResponse),
		},
		ctx:    ctx,
		cancel: cancel,
	}
	// go job.run()
	return job
}

func (job *vframeJob) Cuts() <-chan vframe.CutResponse { return job.chs[0] }
func (job *vframeJob) Error() error                    { return job.err }
func (job *vframeJob) Stop()                           { job.cancel() }

func (job *vframeJob) run() {

	var (
		xl = xlog.FromContextSafe(job.ctx)
	)

	for {
		var end = false
		select {
		case cut, ok := <-job.Job.Cuts():
			if !ok {
				end = true
				break
			}
			for i, ch := range job.chs {
				if job.openChs[i] {
					ch <- cut
				}
			}
		case <-job.ctx.Done():
			xl.Infof("cancel vframe job")
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

func spawnVframeJob(ctx context.Context, _job vframe.Job, n int) []vframe.Job {
	_vjob, _ok := _job.(VframeJob)
	job := newVframeJob(ctx, _job)
	jobs := make([]vframe.Job, 0, n)
	if _ok {
		jobs = append(jobs, &_VframeJob{Job: job, actual: _vjob.Params(), origin: _vjob.OriginParams()})
	} else {
		jobs = append(jobs, job)
	}
	for i := 1; i < n; i++ {
		job.chs = append(job.chs, make(chan vframe.CutResponse))
		job.openChs = append(job.openChs, true)
		if _ok {
			jobs = append(jobs,
				&_VframeJob{
					Job:    newSpawnVframeJob(ctx, job, len(job.chs)-1),
					actual: _vjob.Params(),
					origin: _vjob.OriginParams(),
				})
		} else {
			jobs = append(jobs, newSpawnVframeJob(ctx, job, len(job.chs)-1))
		}
	}
	go job.run()
	return jobs
}

type _SpawnVframeJob struct {
	*vframeJob
	index int
}

func newSpawnVframeJob(
	ctx context.Context, job *vframeJob, index int,
) *_SpawnVframeJob {
	return &_SpawnVframeJob{
		vframeJob: job,
		index:     index,
	}
}

func (job *_SpawnVframeJob) Cuts() <-chan vframe.CutResponse { return job.chs[job.index] }
func (job *_SpawnVframeJob) Error() error                    { return job.vframeJob.Error() }
func (job *_SpawnVframeJob) Stop() {
	job.vframeJob.openChs[job.index] = false
	for { // 消费完channel，防止生产者卡在该ch
		_, ok := <-job.chs[job.index]
		if !ok {
			break
		}
	}
}

////////////////////////////////////////////////////////////////////////////////

/*

----------------------- 2.0 ----------------------- 4.0 ----------------------
-- 0.5 -- 1.0 -- 1.5 -- 2.0 -- 2.5 -- 3.0 -- 3.5 -- 4.0 -- 4.5 -- 5.0 -- 5.5 -
----|-------------|------|------|-------------|------|------|-------------|---
-- 0.0---------------------------2.0 ----------------------- 4.0 ---------------


----------------------- 0.4 ----------------------- 0.8 -----------------------
-- 0.1 -- 0.2 -- 0.3 -- 0.4 -- 0.5 -- 0.6 -- 0.7 -- 0.8 -- 0.9 -- 1.0 -- 1.1 --
------------------|------|------|-------------|------|------|-------------|----
------------------------------ 0.0 ----------------------- 0.4 -----------------

*/
const (
	__FIXED_ZERO   byte = 0x00
	__FIXED_SELECT byte = 0x01
	__FIXED_TARGET byte = 0x02
	__FIXED_ROUND  byte = 0x04
)

type _FixedTimeline struct {
	Interval, OriginInterval int64
	FixedDurations           []int64
	lasts                    []int64
	last                     int64
}

func newFixedTimeline(
	interval, originInterval int64,
	fixedDurations []int64,
) *_FixedTimeline {

	var (
		n     = len(fixedDurations)
		lasts = make([]int64, n)
	)
	for i, d := range fixedDurations {
		for ; d > 0; d -= originInterval {
		}
		lasts[i] = d
	}
	return &_FixedTimeline{
		Interval:       interval,
		OriginInterval: originInterval,
		FixedDurations: fixedDurations,
		lasts:          lasts,
		last:           -1 * interval,
	}
}

func (tl *_FixedTimeline) update(now int64) (isSelect byte) {
	var d = tl.Interval / 2
	for i, last := range tl.lasts {
		for {
			if last+d < now && last+tl.OriginInterval-d < now {
				last = last + tl.OriginInterval
				tl.lasts[i] = last
			} else {
				break
			}
		}
		if last-d < now && now < last+d {
			isSelect |= __FIXED_SELECT
			if tl.FixedDurations[i] == 0 {
				isSelect |= __FIXED_TARGET
			}
			if i == len(tl.lasts)-1 {
				isSelect |= __FIXED_ROUND
			}
		}
	}
	return
}

func (tl *_FixedTimeline) Update(now int64) (
	skip bool,
	misses []struct {
		Offset int64
		Select byte
	},
	isSelect byte,
) {
	var d = tl.Interval / 2
	if now < tl.last+d {
		skip = true
		return
	}
	misses = make([]struct {
		Offset int64
		Select byte
	}, 0)
	for {
		tl.last = tl.last + tl.Interval
		if tl.last+d > now {
			break
		}
		misses = append(misses,
			struct {
				Offset int64
				Select byte
			}{Offset: tl.last, Select: tl.update(tl.last)},
		)
	}
	isSelect = tl.update(now)
	return
}

func (tl *_FixedTimeline) Tail() []struct {
	Offset int64
	Select byte
} {
	var last = tl.last + tl.FixedDurations[len(tl.FixedDurations)-1]
	_, misses, isSelect := tl.Update(last)
	return append(misses,
		struct {
			Offset int64
			Select byte
		}{Offset: last, Select: isSelect})
}
