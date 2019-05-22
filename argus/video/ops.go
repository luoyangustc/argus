package video

import (
	"context"
	"strings"
	"sync"

	"qiniu.com/argus/video/vframe"
)

const (
	LABEL_SELECT_INGORE      = 0x01
	LABEL_SELECT_CHOOSE_ONLY = 0x02

	TERMINATE_MODE_CUT     = 0x01
	TERMINATE_MODE_SEGMENT = 0x02

	MAX_OP_COUNT = 100
)

type OPEnv struct {
	Uid   uint32
	Utype uint32
}

type OPs interface {
	Load() map[string]OP
	ResetOP(string, *OPConfig)
	Create(context.Context, map[string]OPParams, OPEnv) (map[string]OP, bool)
}

type OP interface {
	Fork(context.Context, OPParams, OPEnv) (OP, bool)
	Count() int32
	Reset(context.Context) error
}

type CutOP interface {
	// NewCuts(context.Context, vframe.Job) Cuts
	NewCuts(context.Context, *vframe.VframeParams, *vframe.VframeParams) CutsV2
}
type SpecialCutOP interface {
	VframeParams(context.Context, vframe.VframeParams) *vframe.VframeParams
}

type ClipOP interface {
	NewClips(context.Context, vframe.Job) Clips
}

////////////////////////////////////////////////////////////////////////////////

type NewOP func(OPConfig) OP

var (
	_DefaultFactory = map[string]NewOP{}
)

func RegisterOP(name string, newo NewOP) {
	_DefaultFactory[name] = newo
}

//----------------------------------------------------------------------------//

type ops struct {
	ops map[string]OP
	*sync.RWMutex
}

func NewOPsDirect(ops_ map[string]OP) OPs { return &ops{ops: ops_, RWMutex: new(sync.RWMutex)} }
func NewOPs(configs map[string]OPConfig) OPs {
	_ops := &ops{
		ops:     make(map[string]OP),
		RWMutex: new(sync.RWMutex),
	}
	for name, config := range configs {
		_new, ok := _DefaultFactory[name]
		if !ok {
			continue
		}
		_ops.ops[name] = _new(config)
	}

	return _ops
}

func (s *ops) Load() map[string]OP {
	s.Lock()
	defer s.Unlock()
	return s.ops
}

func (s *ops) ResetOP(name string, config *OPConfig) {

	s.Lock()
	defer s.Unlock()

	if config == nil {
		delete(s.ops, name)
	} else {
		_new, ok := _DefaultFactory[name]
		if ok {
			s.ops[name] = _new(*config)
		}
	}

}

func (s *ops) Create(ctx context.Context, params map[string]OPParams, env OPEnv) (
	map[string]OP, bool) {

	s.RLock()
	defer s.RUnlock()

	var (
		_ops = make(map[string]OP)
		ok   = true
	)
	defer func() {
		if !ok {
			for _, op := range _ops {
				if op != nil {
					_ = op.Reset(ctx)
				}
			}
		}
	}()

	for name, _params := range params {
		var op OP
		if op, ok = s.ops[name]; !ok {
			return nil, ok
		}
		if _ops[name], ok = op.Fork(ctx, _params, env); !ok {
			return nil, ok
		}
	}
	return _ops, true
}

////////////////////////////////////////////////////////////////////////////////

// OPMeta OP处理过程结果的数据结构
// Sample:
//	meta := new(OPMeta)
//	for cResp := make(chan vframe.CutResponse) {
//		meta.Append(ctx, cResp)
//		if meta.End(ctx) {
//			break
//		}
//	}
//	result := meta.Result(ctx)
type OPMeta interface {
	// Append(context.Context, interface{}) bool // true indicates selected, otherwise it's discarded.
	Result(context.Context) EndResult
	End(context.Context) bool
}

type CutOPMeta interface {
	OPMeta
	CutSelect(CutResultWithLabels) (bool, CutResultWithLabels)
	Append(context.Context, CutResultWithLabels) bool
}

type ClipOPMeta interface {
	OPMeta
	Append(context.Context, _ClipResult) bool
}

//----------------------------------------------------------------------------//
var _ OPMeta = (*simpleOPMetaV2)(nil)

type simpleOPMetaV2 struct {
	SegmentParams
	OPParams

	SegmentResult
	EndResult

	onlySomeLabel bool
	labelsCount   map[string]int
}

func newSimpleOPMetaV2(
	segmentParams SegmentParams, params OPParams,
) *simpleOPMetaV2 {
	var onlySomeLabel = false
	for _, label := range params.Labels {
		if label.Select == LABEL_SELECT_CHOOSE_ONLY {
			onlySomeLabel = true
			break
		}
	}
	meta := &simpleOPMetaV2{
		SegmentParams: segmentParams,
		OPParams:      params,
		onlySomeLabel: onlySomeLabel,
		labelsCount:   make(map[string]int),
	}
	return meta
}

func (meta *simpleOPMetaV2) appendLabel(
	labels []ResultLabel, label string, score float32,
) []ResultLabel {
	for j, _label := range labels {
		if _label.Name != label {
			continue
		}
		if score > _label.Score {
			labels[j].Score = score
		}
		return labels
	}
	return append(labels, ResultLabel{Name: label, Score: score})
}

func (meta *simpleOPMetaV2) isSelect(ctx context.Context, labels []ResultLabel) (bool, []ResultLabel) {
	is := func(label string, score float32) bool {
		if meta.onlySomeLabel {
			for _, labelParam := range meta.OPParams.Labels {
				if labelParam.Select == LABEL_SELECT_CHOOSE_ONLY && label == labelParam.Name {
					return score >= labelParam.Score
				}
			}
			return false
		}
		for _, labelParam := range meta.OPParams.Labels {
			if labelParam.Select == LABEL_SELECT_INGORE && label == labelParam.Name {
				return score >= labelParam.Score
			}
		}
		return true
	}
	var ret = make([]ResultLabel, 0, len(labels))
	for _, label := range labels {
		if is(label.Name, label.Score) {
			ret = append(ret, label)
		}
	}
	return len(ret) > 0, ret
}

func (meta *simpleOPMetaV2) Append(
	ctx context.Context, clip _ClipResult,
) bool {

	meta.SegmentResult = SegmentResult{
		OffsetBegin: clip.OffsetBegin,
		OffsetEnd:   clip.OffsetEnd,
		Labels:      []ResultLabel{clip.Labels[0]},
		Clips:       []ClipResult{clip.ClipResult},
	}

	//每个segments的label存储score最大的
	for i := 1; i < len(clip.Labels); i++ {
		if clip.Labels[i].Score > meta.SegmentResult.Labels[0].Score {
			meta.SegmentResult.Labels[0] = clip.Labels[i]
		}
	}

	if len(meta.EndResult.Result.Segments) == 0 {
		meta.EndResult.Result.Segments = []SegmentResult{meta.SegmentResult}
	} else {
		n := len(meta.EndResult.Result.Segments)
		if clip.OffsetBegin < meta.EndResult.Result.Segments[n-1].OffsetEnd && strings.Compare(meta.EndResult.Result.Segments[n-1].Labels[0].Name, meta.SegmentResult.Labels[0].Name) == 0 {
			meta.EndResult.Result.Segments[n-1].OffsetEnd = clip.OffsetEnd
			clen1 := float32(len(meta.EndResult.Result.Segments[n-1].Clips))
			clen2 := float32(len(meta.SegmentResult.Clips))
			meta.EndResult.Result.Segments[n-1].Clips = append(meta.EndResult.Result.Segments[n-1].Clips, clip.ClipResult)

			meta.EndResult.Result.Segments[n-1].Labels[0].Score = (meta.EndResult.Result.Segments[n-1].Labels[0].Score*clen1 + meta.SegmentResult.Labels[0].Score*clen2) / (clen1 + clen2)
		} else {
			meta.EndResult.Result.Segments = append(meta.EndResult.Result.Segments,
				SegmentResult{
					OffsetBegin: clip.OffsetBegin,
					OffsetEnd:   clip.OffsetEnd,
					Labels:      clip.Labels,
					Clips:       []ClipResult{clip.ClipResult},
				},
			)
		}
	}
	return true
}

func (meta *simpleOPMetaV2) Result(ctx context.Context) EndResult {
	return meta.EndResult
}

func (meta *simpleOPMetaV2) End(context.Context) bool {
	if meta.OPParams.Terminate.Mode == TERMINATE_MODE_CUT || meta.OPParams.Terminate.Mode == TERMINATE_MODE_SEGMENT {
		for label, max := range meta.OPParams.Terminate.Labels {
			if count, ok := meta.labelsCount[label]; ok && count >= max {
				return true
			}
		}
	}
	return false
}

////////////////////////////////////////////////////////////////////////////////

var _ = func() bool {
	_DefaultFactory["foo"] = func(config OPConfig) OP {
		op := eFoo{
			OPConfig: config,
		}
		if len(config.Instances) > 0 {
			op.pool = make(chan eFoo, len(config.Instances))
			for _, host := range config.Instances {
				op.pool <- eFoo{
					OPConfig: OPConfig{
						Host: host,
					},
					pool: op.pool,
				}
			}
		}
		return op
	}
	return true
}()

type eFoo struct {
	OPConfig
	OPEnv
	pool chan eFoo
}

func (e eFoo) Fork(ctx context.Context, params OPParams, env OPEnv) (OP, bool) {

	if e.pool != nil {
		select {
		case op := <-e.pool:
			op.OPConfig.Params = params
			op.OPEnv = env
			return op, true
		default:
			return nil, false
		}
	}
	return eFoo{
		OPConfig: OPConfig{
			Host:   e.OPConfig.Host,
			Params: params,
		},
		OPEnv: env,
	}, true
}

func (e eFoo) Eval(ctx context.Context, uris []string) ([]interface{}, error) {
	return make([]interface{}, len(uris)), nil
}

func (e eFoo) Reset(ctx context.Context) error {
	if e.pool != nil {
		e.pool <- e
	}
	return nil
}

func (e eFoo) Count() int32 {
	if e.pool != nil {
		return int32(len(e.pool))
	}
	return MAX_OP_COUNT
}

func (e eFoo) NewCuts(ctx context.Context, params, originParams *vframe.VframeParams) CutsV2 {
	return NewSimpleCutsV2(params, originParams,
		func(ctx context.Context, uris []string) ([]interface{}, error) {
			return e.Eval(ctx, uris)
		},
		func(ctx context.Context, v interface{}) ([]string, []float32, error) {
			return []string{"foo"}, []float32{1.0}, nil
		},
	)
}
