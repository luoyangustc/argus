package vod

import (
	"context"
	"path"

	xlog "github.com/qiniu/xlog.v1"

	"qiniu.com/argus/sdk/video"
	. "qiniu.com/argus/service/service"
	video0 "qiniu.com/argus/video"
	"qiniu.com/argus/video/vframe"
)

var _ Service = service{}

type Service interface {
	Video(context.Context, video0.VideoRequest) (interface{}, error)
}

type service struct {
	DefaultVframeParams vframe.VframeParams
	uriProxy            vframe.URIProxy
	Workerspace         string
	OPs
}

func NewService(
	ctx context.Context,
	vframeParams vframe.VframeParams,
	uriProxy vframe.URIProxy,
	workerspace string,
	ops OPs,
) service {
	return service{
		DefaultVframeParams: vframeParams,
		uriProxy:            uriProxy,
		Workerspace:         workerspace,
		OPs:                 ops,
	}
}

func (s service) video(ctx context.Context, req video0.VideoRequest) (
	map[string]struct {
		Labels   []video0.ResultLabel   `json:"labels"`
		Segments []video0.SegmentResult `json:"segments"`
	},
	error) {
	var (
		vframeReq vframe.VframeRequest
	)

	{
		if req.Params.Vframe == nil || req.Params.Vframe.Mode == nil {
			req.Params.Vframe = &s.DefaultVframeParams
		}
		if req.Params.Vframe.GetMode() == vframe.MODE_INTERVAL &&
			req.Params.Vframe.Interval == 0 {
			req.Params.Vframe.Interval = vframe.DEFAULT_INTERVAL
		}
		vframeReq.Data.URI = req.Data.URI
		if s.uriProxy != nil {
			vframeReq.Data.URI = s.uriProxy.URI(vframeReq.Data.URI)
		}
		vframeReq.Params = *req.Params.Vframe
		if req.Params.SegmentParams == nil {
			req.Params.SegmentParams = &video0.SegmentParams{}
		}
	}
	{
		if req.Params.Vframe.GetMode() != vframe.MODE_INTERVAL &&
			req.Params.Vframe.GetMode() != vframe.MODE_KEY {
			return nil, ErrArgs("invalid mode, allow mode is [0, 1]")
		}
		if req.Params.Vframe.GetMode() == vframe.MODE_INTERVAL {
			if req.Params.Vframe.Interval < 0 || req.Params.Vframe.Interval > 10 {
				return nil, ErrArgs("invalid interval, allow interval is [0, 10]")
			}
		}
	}

	var (
		ops = make(map[string]struct {
			CutOP
			Meta video0.CutOPMeta
		})
		pipes = make(map[string]video.CutsPipe)
	)
	for _, op := range req.Ops {
		of, ok := s.OPs[op.OP]
		if !ok {
			return nil, ErrArgs("bad op")
		}
		_op, err := of.Create(ctx, op.Params)
		if err != nil {
			return nil, err
		}
		if cutOP, ok1 := _op.(CutOP); ok1 {
			if __op, ok2 := _op.(SpecialCutOP); ok2 {
				if _params := __op.VframeParams(ctx, vframeReq.Params); _params != nil {
					vframeReq.Params = *_params
				}
			}
			ops[op.OP] = struct {
				CutOP
				Meta video0.CutOPMeta
			}{
				CutOP: cutOP,
				Meta: video0.NewSimpleCutOPMeta(
					*req.Params.SegmentParams,
					cutOP.Params(),
					nil, nil, true,
				),
			}
		}
	}
	options := make([]video.CutOpOption, 0)
	if req.Params.Vframe.GetMode() == vframe.MODE_INTERVAL {
		options = append(options,
			video.WithInterval(
				int64(vframeReq.Params.Interval*1000),
				int64(vframeReq.Params.Interval*1000)/2),
			video.WithCutFilter(func(offsetMS int64) bool {
				return offsetMS%(int64(req.Params.Vframe.Interval*1000)) == 0
			}),
		)
	}
	var err error
	for name, op := range ops {
		pipes[name], err = op.CutOP.NewCuts(ctx, *req.Params.Vframe, options...)
		if err != nil {
			return nil, formatError(err)
		}
	}

	vf := video.NewGenpic(ctx,
		path.Join(s.Workerspace, xlog.GenReqId()),
		vframeReq.Data.URI,
		vframeReq.Params.GetMode(),
		int64(vframeReq.Params.Interval*1000),
		0, 0, false,
	)
	defer vf.Close()
	resps, err := video.VideoEnd(ctx, vf, video.CreateMultiCutsPipe(pipes))
	if err != nil {
		return nil, formatError(err)
	}

	for _, resp := range resps {
		rets_ := resp.Result.(map[string]video.CutResponse)
		for name, ret := range rets_ {
			if ret.Error != nil {
				// TODO
				continue
			}
			if ops[name].Meta.End(ctx) { // 先简化提前退出功能
				continue
			}
			ret1 := ret.Result.(video0.CutResultWithLabels)
			ret1.CutResult.Offset = ret.OffsetMS
			ops[name].Meta.Append(ctx, ret1)
		}
	}
	rets := make(map[string]struct {
		Labels   []video0.ResultLabel   `json:"labels"`
		Segments []video0.SegmentResult `json:"segments"`
	})
	for name, op := range ops {
		rets[name] = op.Meta.Result(ctx).Result
	}
	return rets, nil
}

func (s service) Video(ctx context.Context, req video0.VideoRequest) (interface{}, error) {

	var (
		id = req.CmdArgs[0] // ID
		xl = xlog.FromContextSafe(ctx)
	)
	xl.Infof("video begin: %v", req.Data.URI)

	if req.Ops == nil || len(req.Ops) == 0 {
		xl.Warnf("Empty OP Request. %s", id)
		return nil, ErrArgs("empty op")
	}

	for _, op := range req.Ops {
		if _, ok := s.OPs[op.OP]; !ok {
			xl.Warnf("Bad OP Request: %#v %#v", s.OPs, op)
			return nil, ErrArgs("bad op")
		}
	}

	ret, err := s.video(ctx, req)
	xl.Infof("video end: %v %v", req.Data.URI, err)
	return ret, err
}
