package video

import (
	"context"
)

func SelectCut(
	labels []struct {
		Name   string  `json:"label"`
		Select int     `json:"select"`
		Score  float32 `json:"score"`
	},
	ignoreEmptyLabels bool,
	ret CutResultWithLabels) (bool, CutResultWithLabels) {
	return selectCut(labels, ignoreEmptyLabels)(ret)
}

func selectCut(labels []struct {
	Name   string  `json:"label"`
	Select int     `json:"select"`
	Score  float32 `json:"score"`
}, ignoreEmptyLabels bool) func(CutResultWithLabels) (bool, CutResultWithLabels) {

	var onlySomeLabel = false
	for _, label := range labels {
		if label.Select == LABEL_SELECT_CHOOSE_ONLY {
			onlySomeLabel = true
			break
		}
	}
	is := func(label string, score float32) bool {
		if onlySomeLabel {
			for _, labelParam := range labels {
				if labelParam.Select == LABEL_SELECT_CHOOSE_ONLY && label == labelParam.Name {
					return score >= labelParam.Score
				}
			}
			return false
		}
		for _, labelParam := range labels {
			if labelParam.Select == LABEL_SELECT_INGORE && label == labelParam.Name {
				var min float32 = 0.000001
				if (labelParam.Score-0.0) < min && (0.0-labelParam.Score) < min {
					labelParam.Score = 1.001
				}
				return score > labelParam.Score
			}
		}
		return true
	}

	return func(cut CutResultWithLabels) (bool, CutResultWithLabels) {
		var ret = make([]ResultLabel, 0, len(cut.Labels))

		if len(cut.Labels) == 0 && !ignoreEmptyLabels {
			return true, CutResultWithLabels{CutResult: cut.CutResult, Labels: ret}
		}

		for _, label := range cut.Labels {
			if is(label.Name, label.Score) {
				ret = append(ret, label)
			}
		}
		// xlog.NewDummy().Infof("%#v %v", ret, cut.CutResult.Result)
		return len(ret) > 0, CutResultWithLabels{CutResult: cut.CutResult, Labels: ret}
	}
}

//----------------------------------------------------------------------------//

func mergeLabels(
	labels []ResultLabel, labels2 ...ResultLabel,
) []ResultLabel {
	for _, _label2 := range labels2 {
		var found bool
		for j, _label := range labels {
			if _label.Name != _label2.Name {
				continue
			}
			if _label2.Score > _label.Score {
				labels[j].Score = _label2.Score
			}
			found = true
			break
		}
		if !found {
			labels = append(labels,
				ResultLabel{Name: _label2.Name, Score: _label2.Score})
		}
	}
	return labels
}

type SegmentMeta interface {
	Interrupt()
	Append(CutResultWithLabels)
	Result() *SegmentResult
	Next() (SegmentMeta, *SegmentResult)
}

var _ SegmentMeta = &_AutoSegmentMeta{}

type _AutoSegmentMeta struct {
	*SegmentResult
	hasNext bool
	next    *SegmentResult
}

func (meta _AutoSegmentMeta) canAppend(labels []ResultLabel) bool {
	if len(labels) != len(meta.Labels) {
		return false
	}

	var found bool
	for _, label := range labels {
		found = false
		for _, slabel := range meta.Labels {
			if label.Name == slabel.Name {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}
	return true
}

func (meta *_AutoSegmentMeta) Append(cut CutResultWithLabels) {
	if meta.SegmentResult != nil && meta.canAppend(cut.Labels) {
		if meta.Cuts == nil || len(meta.Cuts) == 0 {
			meta.OffsetBegin = cut.Offset
			meta.Cuts = make([]CutResult, 0)
		}
		meta.Cuts = append(meta.Cuts, cut.CutResult)
		meta.OffsetEnd = cut.Offset
		meta.Labels = mergeLabels(meta.Labels, cut.Labels...)
	} else {
		meta.hasNext, meta.next = true, &SegmentResult{
			OffsetBegin: cut.Offset,
			OffsetEnd:   cut.Offset,
			Labels:      cut.Labels,
			Cuts:        []CutResult{cut.CutResult},
		}
	}
}

func (meta *_AutoSegmentMeta) Interrupt()            { meta.hasNext = true }
func (meta _AutoSegmentMeta) Result() *SegmentResult { return meta.SegmentResult }
func (meta _AutoSegmentMeta) Next() (SegmentMeta, *SegmentResult) {
	if !meta.hasNext {
		return nil, meta.SegmentResult
	}
	return &_AutoSegmentMeta{SegmentResult: meta.next}, meta.SegmentResult
}

var _ SegmentMeta = &_StaticSegmentMeta{}

type _StaticSegmentMeta struct {
	Interval int64

	*SegmentResult
	next *SegmentResult
}

func (meta *_StaticSegmentMeta) Append(cut CutResultWithLabels) {
	if meta.SegmentResult != nil &&
		cut.Offset-meta.Interval < meta.SegmentResult.OffsetBegin {
		if meta.Cuts == nil || len(meta.Cuts) == 0 {
			meta.OffsetBegin = cut.Offset
			meta.Cuts = make([]CutResult, 0)
		}
		meta.Cuts = append(meta.Cuts, cut.CutResult)
		meta.OffsetEnd = cut.Offset
		meta.Labels = mergeLabels(meta.Labels, cut.Labels...)
	} else {
		meta.next = &SegmentResult{
			OffsetBegin: cut.Offset,
			OffsetEnd:   cut.Offset,
			Labels:      cut.Labels,
			Cuts:        []CutResult{cut.CutResult},
		}
	}
}

func (meta *_StaticSegmentMeta) Interrupt()            {}
func (meta _StaticSegmentMeta) Result() *SegmentResult { return meta.SegmentResult }
func (meta _StaticSegmentMeta) Next() (SegmentMeta, *SegmentResult) {
	if meta.next == nil {
		return nil, meta.SegmentResult
	}
	return &_StaticSegmentMeta{Interval: meta.Interval, SegmentResult: meta.next},
		meta.SegmentResult
}

//----------------------------------------------------------------------------//

func NewSimpleCutOPMeta(
	segmentParams SegmentParams, params OPParams,
	cutHook CutHook, segmentHook SegmentHook,
	needEndResult bool,
) *simpleCutOPMeta {
	var sm SegmentMeta
	switch segmentParams.Mode {
	case 0:
		sm = &_AutoSegmentMeta{}
	case 1:
		sm = &_StaticSegmentMeta{Interval: int64(segmentParams.Interval)}
	default:
		sm = &_AutoSegmentMeta{}
	}
	var end *EndResult
	if needEndResult {
		end = &EndResult{}
	}
	meta := &simpleCutOPMeta{
		SegmentParams: segmentParams,
		OPParams:      params,

		SegmentMeta: sm,
		cutSelect:   selectCut(params.Labels, params.IgnoreEmptyLabels),

		labelsCount: make(map[string]int),

		CutHook:     cutHook,
		SegmentHook: segmentHook,
		EndResult:   end,
	}
	return meta
}

var _ CutOPMeta = (*simpleCutOPMeta)(nil)

type simpleCutOPMeta struct {
	SegmentParams
	OPParams

	SegmentMeta
	*EndResult

	cutSelect   func(CutResultWithLabels) (bool, CutResultWithLabels)
	labelsCount map[string]int

	CutHook     CutHook
	SegmentHook SegmentHook
}

func (meta *simpleCutOPMeta) CutSelect(cut CutResultWithLabels) (bool, CutResultWithLabels) {
	return meta.cutSelect(cut)
}

func (meta *simpleCutOPMeta) appendSegment(end EndResult, segment SegmentResult) EndResult {
	if end.Result.Segments == nil || len(end.Result.Segments) == 0 {
		end.OffsetBegin = segment.OffsetBegin
		end.Result.Segments = []SegmentResult{}
	}
	if end.Result.Labels == nil || len(end.Result.Labels) == 0 {
		end.Result.Labels = []ResultLabel{}
	}
	end.Result.Segments = append(end.Result.Segments, segment)
	end.Result.Labels = mergeLabels(end.Result.Labels, segment.Labels...)
	end.OffsetEnd = segment.OffsetEnd
	return end
}

func (meta *simpleCutOPMeta) Append(
	ctx context.Context, cut CutResultWithLabels,
) bool {

	var (
		// xl       = xlog.FromContextSafe(ctx)
		selected bool
	)

	selected, cut = meta.CutSelect(cut)
	if !selected {
		meta.SegmentMeta.Interrupt()
		// return false
	} else {
		if meta.CutHook != nil {
			_ = meta.CutHook.Cut(ctx, cut.CutResult)
		}
		meta.SegmentMeta.Append(cut)
	}

	var (
		nw bool
		sm SegmentMeta
		sr *SegmentResult
	)
	if sm, sr = meta.SegmentMeta.Next(); sm != nil {
		if sr != nil {
			if meta.SegmentHook != nil {
				_ = meta.SegmentHook.Segment(ctx, *sr)
			}
			if meta.EndResult != nil {
				end := meta.appendSegment(*meta.EndResult, *sr)
				meta.EndResult = &end
			}
		}
		meta.SegmentMeta = sm
		nw = true
	}

	switch meta.OPParams.Terminate.Mode {
	case TERMINATE_MODE_CUT:
		for _, label := range cut.Labels {
			if _, ok := meta.labelsCount[label.Name]; ok {
				meta.labelsCount[label.Name]++
			} else {
				meta.labelsCount[label.Name] = 1
			}
		}
	case TERMINATE_MODE_SEGMENT:
		if nw && sr != nil {
			for _, label := range sr.Labels {
				if _, ok := meta.labelsCount[label.Name]; ok {
					meta.labelsCount[label.Name]++
				} else {
					meta.labelsCount[label.Name] = 1
				}
			}
		}
	}
	// xl.Infof("%#v %#v", meta.SegmentMeta, meta.EndResult)

	return true
}

func (meta *simpleCutOPMeta) Result(context.Context) EndResult {
	if meta.EndResult == nil {
		return EndResult{}
	}
	if sr := meta.SegmentMeta.Result(); sr != nil {
		return meta.appendSegment(*meta.EndResult, *sr)
	}
	return *meta.EndResult
}

func (meta *simpleCutOPMeta) End(context.Context) bool {
	if meta.OPParams.Terminate.Mode == TERMINATE_MODE_CUT || meta.OPParams.Terminate.Mode == TERMINATE_MODE_SEGMENT {
		for label, max := range meta.OPParams.Terminate.Labels {
			if count, ok := meta.labelsCount[label]; ok && count >= max {
				return true
			}
		}
	}
	return false
}
