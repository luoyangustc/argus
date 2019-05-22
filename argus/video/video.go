package video

import (
	"context"
	"sync"

	"github.com/qiniu/xlog.v1"
	"qiniu.com/argus/video/segment"
	"qiniu.com/argus/video/vframe"
)

const (
	DEFAULT_INTERVAL = 5
	DEFAULT_DURATION = 10
)

type Video interface {
	Run(
		context.Context,
		VideoRequest,
		map[string]OP,
		SaverOPHook,
		func(string) EndHook,
		func(string) CutHook,
		func(string) SegmentHook,
	) error
}

type video struct {
	vframe.Vframe
	vframe.VframeParams

	segment.Segment
	segment.SegmentParams
}

func NewVideo(
	_vframe vframe.Vframe,
	defaultVframeParams vframe.VframeParams,
	_segment segment.Segment,
	defaultSegmentParams segment.SegmentParams,
) Video {
	return video{
		Vframe: _vframe, VframeParams: defaultVframeParams,
		Segment: _segment, SegmentParams: defaultSegmentParams,
	}
}

func GenVframeRequest(
	ctx context.Context, req VideoRequest,
	ops map[string]OP, defaultVframeParams *vframe.VframeParams,
) (vframe.VframeRequest, *vframe.VframeParams, *vframe.VframeParams, []int, []OP) {
	var (
		originParams *vframe.VframeParams
		params       *vframe.VframeParams
		vframeReq    = vframe.VframeRequest{}
		indexes      = make([]int, 0, len(req.Ops))
		_ops         = make([]OP, 0, len(req.Ops))
	)

	vframeReq.Data.URI = req.Data.URI
	if req.Params.Vframe == nil || req.Params.Vframe.Mode == nil {
		originParams = defaultVframeParams
	} else {
		originParams = req.Params.Vframe
	}

	if originParams.GetMode() == vframe.MODE_INTERVAL && originParams.Interval == 0 {
		originParams.Interval = vframe.DEFAULT_INTERVAL
	}
	params = originParams

	for i, op := range req.Ops {
		_op := ops[op.OP]
		if _, ok := _op.(CutOP); ok {
			if __op, _ok := _op.(SpecialCutOP); _ok {
				if _params := __op.VframeParams(ctx, *params); _params != nil {
					params = _params
				}
			}
			indexes = append(indexes, i)
			_ops = append(_ops, _op)
		}
	}
	vframeReq.Params = *params

	if req.Params.Live != nil && req.Params.Live.Timeout > 0 {
		vframeReq.Live = &vframe.LiveParams{
			Timeout:   req.Params.Live.Timeout,
			Downsteam: req.Params.Live.Downstream,
		}
	}

	return vframeReq, originParams, params, indexes, _ops
}

func (v video) Run(
	ctx context.Context,
	req VideoRequest,
	ops map[string]OP,
	saverOPHook SaverOPHook,
	ends func(string) EndHook,
	cuts func(string) CutHook,
	segments func(string) SegmentHook, // TODO
) error {

	var (
		xl     = xlog.FromContextSafe(ctx)
		wait   = new(sync.WaitGroup)
		errors = make(chan error, len(req.Ops)*2)
	)

	{
		vframeReq, originParams, params, indexes, _ops := GenVframeRequest(ctx, req, ops, &v.VframeParams)

		if len(indexes) > 0 {
			vJob, _ := v.Vframe.Run(ctx, vframeReq) // TODO VframeJob
			if params != originParams {
				vJob = &_VframeJob{Job: vJob, actual: *params, origin: *originParams}
			}
			jobs := spawnVframeJob(ctx, vJob, len(indexes))

			for i, j := range indexes {
				var (
					op    = req.Ops[j]
					_op   = _ops[i]
					hook  = ends(op.OP)
					saver Saver
				)
				if saverOPHook != nil {
					saver, _ = saverOPHook.Get(ctx, op.OP)
				}

				xl.Infof("op: %s %#v", op.OP, _op)

				wait.Add(1)
				go func(
					ctx context.Context, job vframe.Job, runner _VideoCutsRunner,
					segmentParams *SegmentParams, opParams OPParams,
				) {
					defer wait.Done()
					var sp SegmentParams
					if segmentParams != nil {
						sp = *segmentParams
					}
					meta := NewSimpleCutOPMeta(sp, opParams, nil, nil, true)
					err := runner.RunOP(ctx, job, _op.(CutOP), meta, hook, saver)
					if err != nil {
						errors <- err
					}
				}(spawnContext(ctx), jobs[i], _VideoRunnerV1{}, req.Params.SegmentParams, op.Params) // TODO
			}
		}

	}
	{
		var (
			vframeReq = vframe.VframeRequest{}
			indexes   = make([]int, 0, len(req.Ops))
			_ops      = make([]OP, 0, len(req.Ops))
		)

		vframeReq.Data.URI = req.Data.URI
		vframeReq.Params = v.VframeParams
		if req.Params.Vframe != nil {
			if req.Params.Vframe.GetMode() == vframe.MODE_INTERVAL && req.Params.Vframe.Interval != 0 {
				vframeReq.Params.Interval = req.Params.Vframe.Interval
			} else {
				vframeReq.Params.Interval = vframe.VIDEO_CLASSIFY_DEFAULT_INTERVAL
			}
		}
		var mode = vframe.MODE_INTERVAL
		vframeReq.Params.Mode = &mode

		for i, op := range req.Ops {
			_op := ops[op.OP]
			if _, ok := _op.(ClipOP); ok {
				indexes = append(indexes, i)
				_ops = append(_ops, _op)
			}
		}

		if len(indexes) > 0 {
			vJob, _ := v.Vframe.Run(ctx, vframeReq)
			vJob = &_VframeJob{Job: vJob, actual: vframeReq.Params, origin: vframeReq.Params}
			jobs := spawnVframeJob(ctx, vJob, len(indexes))

			for i, j := range indexes {
				var (
					op   = req.Ops[j]
					_op  = _ops[i]
					hook = ends(op.OP)
				)

				xl.Infof("op: %s %#v", op.OP, _op)

				wait.Add(1)
				go func(
					ctx context.Context, job vframe.Job, runner _VideoClipsRunner,
					segmentParams *SegmentParams, opParams OPParams,
				) {
					defer wait.Done()
					var sp SegmentParams
					if segmentParams != nil {
						sp = *segmentParams
					}
					meta := newSimpleOPMetaV2(sp, opParams)
					err := runner.RunOP(ctx, job, _op.(ClipOP), meta, hook)
					if err != nil {
						errors <- err
					}
				}(spawnContext(ctx), jobs[i], _VideoRunnerV2{}, req.Params.SegmentParams, op.Params) // TODO
			}
		}

	}

	wait.Wait()

	select {
	case err := <-errors:
		return err
	default:
		return nil
	}

	// return nil
}

//----------------------------------------------------------------------------//

type _VideoCutsRunner interface {
	RunOP(context.Context, vframe.Job, CutOP, CutOPMeta, EndHook, Saver) error
}

type _VideoRunnerV1 struct{}

func (r _VideoRunnerV1) RunOP(
	ctx context.Context,
	job vframe.Job, op CutOP, meta CutOPMeta,
	hook EndHook,
	saver Saver,
) error {
	var (
		xl   = xlog.FromContextSafe(ctx)
		cuts Cuts // NewCuts(job, op.NewCuts(ctx, job))
		err  error
	)
	if _job, ok := job.(VframeJob); ok {
		params, originParams := _job.Params(), _job.OriginParams()
		cuts = NewCuts(job, op.NewCuts(ctx, &params, &originParams))
	} else {
		cuts = NewCuts(job, op.NewCuts(ctx, nil, nil))
	}

	for {
		cut, ok := cuts.Next(ctx)
		if !ok {
			job.Stop() // TODO
			break
		}
		xl.Debugf("cut. %d", cut.Offset)
		{
			selected, _ := meta.CutSelect(cut)
			if selected {
				if saver != nil {
					if ret, err := saver.Save(ctx, cut.Offset, cut.URI); err != nil {
						xl.Warnf("save failed: %d %s %v", cut.Offset, cut.URI, err)
						cut.URI = ""
					} else {
						cut.URI = ret
					}
				} else {
					cut.URI = "" // 没有另存操作，不返回该字段内容
				}
			} else {
				cut.URI = ""
			}
			meta.Append(ctx, cut)
		}
		// TODO
		if meta.End(ctx) {
			job.Stop()
			break
		}
	}
	if err == nil {
		err = cuts.Error()
		if err != nil {
			// TODO
		}
	}

	endResult := meta.Result(ctx)
	hook.End(ctx, endResult)
	return err
}

//----------------------------------------------------------------------------//

type _VideoClipsRunner interface {
	RunOP(context.Context, vframe.Job, ClipOP, ClipOPMeta, EndHook) error
}

type _VideoRunnerV2 struct{}

func (r _VideoRunnerV2) RunOP(
	ctx context.Context,
	job vframe.Job, op ClipOP, meta ClipOPMeta,
	hook EndHook,
) error {

	var (
		// xl   = xlog.FromContextSafe(ctx)
		clips = op.NewClips(ctx, job)
		err   error
	)

	for {
		clip, ok := clips.Next(ctx)
		if !ok {
			job.Stop() // TODO
			break
		}
		meta.Append(ctx, clip)
		// TODO
		if meta.End(ctx) {
			job.Stop()
			// break
		}
	}
	if err == nil {
		err = clips.Error()
		if err != nil {
			// TODO
		}
	}

	endResult := meta.Result(ctx)
	hook.End(ctx, endResult)
	return err
}

//----------------------------------------------------------------------------//

type CutHook interface {
	Cut(context.Context, CutResult) error
}

type cutHook struct {
	f func(context.Context, CutResult) error
}

func (hook cutHook) Cut(ctx context.Context, ret CutResult) error  { return hook.f(ctx, ret) }
func CutHookFunc(f func(context.Context, CutResult) error) CutHook { return cutHook{f: f} }

type SegmentHook interface {
	Segment(context.Context, SegmentResult) error
}

type segmentHook struct {
	f func(context.Context, SegmentResult) error
}

func (hook segmentHook) Segment(ctx context.Context, ret SegmentResult) error { return hook.f(ctx, ret) }
func segmentHookFunc(f func(context.Context, SegmentResult) error) SegmentHook {
	return segmentHook{f: f}
}

type EndHook interface {
	End(context.Context, EndResult) error
}

type endHook struct {
	f func(context.Context, EndResult) error
}

func (hook endHook) End(ctx context.Context, rest EndResult) error {
	return hook.f(ctx, rest)
}

func EndHookFunc(f func(context.Context, EndResult) error) EndHook {
	return endHook{f: f}
}
