package video

import (
	"context"
	"sync"

	"github.com/qiniu/xlog.v1"
	"qiniu.com/argus/video/segment"
	"qiniu.com/argus/video/vframe"
)

type live struct {
	vframe.Vframe
	vframe.VframeParams

	segment.Segment
	segment.SegmentParams
}

func NewLive(
	_vframe vframe.Vframe,
	defaultVframeParams vframe.VframeParams,
	_segment segment.Segment,
	defaultSegmentParams segment.SegmentParams,
) Video {
	return live{
		Vframe: _vframe, VframeParams: defaultVframeParams,
		Segment: _segment, SegmentParams: defaultSegmentParams,
	}
}

func (v live) Run(
	ctx context.Context,
	req VideoRequest,
	ops map[string]OP,
	saverOPHook SaverOPHook,
	ends func(string) EndHook,
	cuts func(string) CutHook,
	segments func(string) SegmentHook,
) error {

	var (
		xl     = xlog.FromContextSafe(ctx)
		wait   = new(sync.WaitGroup)
		errors = make(chan error, len(req.Ops)*2)
	)

	{
		var (
			originParams *vframe.VframeParams
			params       *vframe.VframeParams
			vframeReq    = vframe.VframeRequest{}
			indexes      = make([]int, 0, len(req.Ops))
			_ops         = make([]OP, 0, len(req.Ops))
		)

		vframeReq.Data.URI = req.Data.URI
		if req.Params.Vframe == nil {
			originParams = &v.VframeParams
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
			vframeReq.Live = &vframe.LiveParams{Timeout: req.Params.Live.Timeout}
		}

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
					meta := NewSimpleCutOPMeta(sp, opParams, cuts(op.OP), segments(op.OP), false)
					err := runner.RunOP(ctx, job, _op.(CutOP), meta, hook, saver)
					if err != nil {
						errors <- err
					}
				}(spawnContext(ctx), jobs[i], _VideoRunnerV1{}, req.Params.SegmentParams, op.Params) // TODO
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
