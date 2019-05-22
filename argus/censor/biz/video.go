package biz

import (
	"encoding/json"
)

type CutResult struct {
	Offset     int64       `json:"offset"`
	URI        string      `json:"uri,omitempty"`
	Suggestion Suggestion  `json:"suggestion"`
	Result     interface{} `json:"result"`
}

type SegmentResult struct {
	OffsetBegin int64       `json:"offset_begin"`
	OffsetEnd   int64       `json:"offset_end"`
	Suggestion  Suggestion  `json:"suggestion"`
	Cuts        []CutResult `json:"cuts,omitempty"`
}

func (seg SegmentResult) Append(cut CutResult) (SegmentResult, *SegmentResult) {
	if len(seg.Cuts) == 0 {
		return SegmentResult{
			OffsetBegin: cut.Offset,
			OffsetEnd:   cut.Offset,
			Suggestion:  cut.Suggestion,
			Cuts:        []CutResult{cut},
		}, nil
	}
	if cut.Suggestion == seg.Suggestion {
		seg.OffsetEnd = cut.Offset
		seg.Cuts = append(seg.Cuts, cut)
		return seg, nil
	}
	return seg, &SegmentResult{
		OffsetBegin: cut.Offset,
		OffsetEnd:   cut.Offset,
		Suggestion:  cut.Suggestion,
		Cuts:        []CutResult{cut},
	}
}

type VideoSceneResult struct {
	Suggestion Suggestion      `json:"suggestion"`
	Segments   []SegmentResult `json:"segments,omitempty"`
}

func (v VideoSceneResult) Append(seg SegmentResult) VideoSceneResult {
	if len(v.Segments) == 0 {
		v.Suggestion = PASS
	}
	v.Suggestion = v.Suggestion.Update(seg.Suggestion)
	v.Segments = append(v.Segments, seg)
	return v
}

////////////////////////////////////////////////////////////////////////////////

// video.CutResult
type OriginCutResult struct {
	Offset int64           `json:"offset"`
	URI    string          `json:"uri,omitempty"`
	Result json.RawMessage `json:"result,omitempty"`
}

func ParseCutPulpResult(ret OriginCutResult, params PulpThreshold) CutResult {
	var ret0 ImagePulpResp
	_ = json.Unmarshal(ret.Result, &ret0)
	var ret1 CutResult
	ret1.Suggestion, ret1.Result = ParsePulp(ret0, params)
	ret1.Offset, ret1.URI = ret.Offset, ret.URI
	return ret1
}

func ParseCutTerrorResult(ret OriginCutResult, params TerrorThreshold) CutResult {
	var ret0 ImageTerrorResp
	_ = json.Unmarshal(ret.Result, &ret0)
	var ret1 CutResult
	ret1.Suggestion, ret1.Result = ParseTerror(ret0, params)
	ret1.Offset, ret1.URI = ret.Offset, ret.URI
	return ret1
}

func ParseCutPoliticianResult(ret OriginCutResult, params PoliticianThreshold) CutResult {
	var ret0 ImagePoliticianResp
	_ = json.Unmarshal(ret.Result, &ret0)
	var ret1 CutResult
	ret1.Suggestion, ret1.Result = ParsePolitician(ret0, params)
	ret1.Offset, ret1.URI = ret.Offset, ret.URI
	return ret1
}

//----------------------------------------------------------------------------//

type OriginVideoOPResult struct {
	Segments []struct {
		Cuts []OriginCutResult `json:"cuts,omitempty"`
	} `json:"segments"`
}

func ParseOriginVideoOPResult(ret0 OriginVideoOPResult,
	cutParser func(OriginCutResult) CutResult,
) VideoSceneResult {

	var ret = VideoSceneResult{
		Suggestion: PASS, // 明确初始化
	}
	for _, seg0 := range ret0.Segments {
		var seg = SegmentResult{}
		for _, cut0 := range seg0.Cuts {
			cut := cutParser(cut0)
			var newSeg *SegmentResult
			seg, newSeg = seg.Append(cut)
			if newSeg == nil {
				continue
			}
			ret = ret.Append(seg)
			seg = *newSeg
		}
		if len(seg.Cuts) > 0 {
			ret = ret.Append(seg)
		}
	}

	return ret
}
