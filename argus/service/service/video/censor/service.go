package censor

import (
	"context"
	"path"
	"sync"

	xlog "github.com/qiniu/xlog.v1"
	"qiniu.com/argus/com/util"
	"qiniu.com/argus/sdk/video"
	. "qiniu.com/argus/service/service"
	pimage "qiniu.com/argus/service/service/image"
	svideo "qiniu.com/argus/service/service/video"
	"qiniu.com/argus/video/vframe"
)

var _ Service = (*service)(nil)

type Service interface {
	Video(context.Context, VideoCensorReq) (*VideoCensorResp, error)
}

type CutResponseWithUri struct {
	video.CutResponse
	Uri string
}
type service struct {
	DefaultCutParams CutParam
	uriProxy         vframe.URIProxy
	saverHook        svideo.SaverHook
	Workerspace      string
	OPs
}

func NewService(
	ctx context.Context,
	cutParam CutParam,
	saverHook svideo.SaverHook,
	uriProxy vframe.URIProxy,
	workerspace string,
	ops OPs,
) *service {
	return &service{
		DefaultCutParams: cutParam,
		uriProxy:         uriProxy,
		Workerspace:      workerspace,
		OPs:              ops,
		saverHook:        saverHook,
	}
}

func (s *service) Video(ctx context.Context, req VideoCensorReq) (*VideoCensorResp, error) {
	var (
		xl       = xlog.FromContextSafe(ctx)
		vid      = xlog.GenReqId()
		videoUri string
		cutParam *CutParam
	)
	xl.Infof("video begin: %v", req.Data.URI)

	{
		// validate scenes
		if len(req.Params.Scenes) == 0 {
			xl.Warnf("empty scenes")
			return nil, ErrArgs("empty scene")
		}

		for _, scene := range req.Params.Scenes {
			if _, ok := s.OPs[scene]; !ok {
				xl.Warnf("bad scene request : %#v %#v", s.OPs, scene)
				return nil, ErrArgs("bad scene")
			}
		}

		cutParam = req.Params.CutParam
		if cutParam == nil {
			cutParam = &s.DefaultCutParams
		}

		// validate mode
		switch cutParam.Mode {
		case vframe.MODE_INTERVAL, vframe.MODE_KEY:
		default:
			return nil, ErrArgs("invalid mode, allow mode is [0, 1]")
		}

		if cutParam.Mode == vframe.MODE_INTERVAL &&
			cutParam.IntervalMsecs == 0 {
			cutParam.IntervalMsecs = DEFAULT_INTERVAL
		}

		// validate interval
		if cutParam.IntervalMsecs < 1000 || cutParam.IntervalMsecs > 60000 {
			return nil, ErrArgs("invalid interval, allow interval is [1000, 60000]")
		}

		// build video uri
		videoUri = req.Data.URI
		if s.uriProxy != nil {
			videoUri = s.uriProxy.URI(videoUri)
		}
	}

	var (
		ops   = make(map[string]OP)
		pipes = make(map[string]video.CutsPipe)
	)

	// build ops from scenes & op factory
	var finalCutParam CutParam = *cutParam
	for _, scene := range req.Params.Scenes {
		of, ok := s.OPs[scene]
		if !ok {
			return nil, ErrArgs("bad scene")
		}

		_op, err := of.Create(ctx)
		if err != nil {
			return nil, formatError(err)
		}

		if __op, ok2 := _op.(SpecialOP); ok2 {
			if _params := __op.CutParam(ctx, finalCutParam); _params != nil {
				finalCutParam = *_params
			}
		}

		ops[scene] = _op
	}

	// build video options
	options := make([]video.CutOpOption, 0)
	if finalCutParam.Mode == vframe.MODE_INTERVAL {
		options = append(options,
			video.WithInterval(
				finalCutParam.IntervalMsecs,
				finalCutParam.IntervalMsecs/2),
			video.WithCutFilter(func(offsetMS int64) bool {
				return offsetMS%cutParam.IntervalMsecs == 0
			}),
		)
	}

	var err error
	// op cut handler
	for name, op := range ops {
		pipes[name], err = op.NewCuts(ctx, finalCutParam, options...)
		if err != nil {
			return nil, formatError(err)
		}
	}

	// video handler
	vf := video.NewGenpic(ctx,
		path.Join(s.Workerspace, vid),
		videoUri,
		finalCutParam.Mode,
		finalCutParam.IntervalMsecs,
		0, 0, false,
	)

	// get response
	var saver svideo.Saver
	if req.Params.Saver.Save && s.saverHook != nil {
		saver, _ = s.saverHook.Get(ctx, vid, nil)
	}
	resps, err := s.video(ctx, vf, video.CreateMultiCutsPipe(pipes), saver)
	if err != nil {
		return nil, formatError(err)
	}

	// build final response
	ret := &VideoCensorResp{
		Code:    200,
		Message: "OK",
		Result: &CensorResult{
			Suggestion: pimage.PASS,
			Scenes:     make(map[string]*SceneResult),
		},
	}
	for _, resp := range resps {
		cutResp_ := resp.Result.(map[string]video.CutResponse)
		for name, cutResp := range cutResp_ {
			if cutResp.Error != nil {
				return nil, formatError(cutResp.Error)
			}

			cutResp1 := cutResp.Result.(pimage.SceneResult)
			cutRet := CutResult{
				Offset:     cutResp.OffsetMS,
				Uri:        resp.Uri,
				Suggestion: cutResp1.Suggestion,
				Details:    cutResp1.Details,
			}

			if _, ok := ret.Result.Scenes[name]; !ok {
				ret.Result.Scenes[name] = &SceneResult{
					Suggestion: pimage.PASS,
				}
			}
			sr := ret.Result.Scenes[name]
			sr.Cuts = append(sr.Cuts, cutRet)
			sr.Suggestion = sr.Suggestion.Update(cutRet.Suggestion)
		}
	}
	for _, sr := range ret.Result.Scenes {
		ret.Result.Suggestion = ret.Result.Suggestion.Update(sr.Suggestion)
	}

	xl.Infof("video end: %v %v", req.Data.URI, err)
	return ret, err
}

func (s *service) video(ctx context.Context, vframe video.Vframe, pipe video.CutsPipe, saver svideo.Saver) ([]CutResponseWithUri, error) {
	// 清除截帧文件
	defer vframe.Close()

	resps := make([]CutResponseWithUri, 0)
	uris := make(map[int64]string)
	bufsize := 16

	var wg sync.WaitGroup
	var m sync.Mutex
	for {
		// 获得当前已截的帧
		cuts, ok := vframe.Next(ctx, bufsize)
		if !ok {
			break
		}

		if saver != nil {
			// 保存帧
			for i, cut := range cuts {
				wg.Add(1)
				go func(ctx context.Context, cut video.CutRequest) {
					defer wg.Done()
					uri, _ := saver.Save(ctx, cut.OffsetMS, string(pimage.DataURI(cut.Body)))
					m.Lock()
					uris[cut.OffsetMS] = uri
					m.Unlock()
				}(util.SpawnContext2(ctx, i), cut)
			}
		}

		// 帧推理
		for _, resp := range pipe.Append(ctx, cuts...) {
			resps = append(resps, CutResponseWithUri{CutResponse: resp})
		}
	}
	if err := vframe.Error(); err != nil {
		return nil, err
	}

	// 帧推理
	for _, resp := range pipe.End(ctx) {
		resps = append(resps, CutResponseWithUri{CutResponse: resp})
	}
	wg.Wait()

	// 记录帧保存地址
	for i := 0; i < len(resps); i++ {
		resps[i].Uri = uris[resps[i].OffsetMS]
	}

	return resps, nil
}
