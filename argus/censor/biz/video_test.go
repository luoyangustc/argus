package biz

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"

	"qiniu.com/argus/utility/censor"
	"qiniu.com/argus/video"
)

func TestSegmentResult(t *testing.T) {
	s1, s2 := SegmentResult{}.Append(CutResult{Offset: 1, Suggestion: REVIEW})
	assert.Equal(t, int64(1), s1.OffsetBegin)
	assert.Equal(t, int64(1), s1.OffsetEnd)
	assert.Equal(t, REVIEW, s1.Suggestion)
	assert.Equal(t, 1, len(s1.Cuts))
	assert.Nil(t, s2)

	s1, s2 = s1.Append(CutResult{Offset: 2, Suggestion: REVIEW})
	assert.Equal(t, int64(1), s1.OffsetBegin)
	assert.Equal(t, int64(2), s1.OffsetEnd)
	assert.Equal(t, REVIEW, s1.Suggestion)
	assert.Equal(t, 2, len(s1.Cuts))
	assert.Nil(t, s2)

	s1, s2 = s1.Append(CutResult{Offset: 3, Suggestion: BLOCK})
	assert.Equal(t, int64(1), s1.OffsetBegin)
	assert.Equal(t, int64(2), s1.OffsetEnd)
	assert.Equal(t, REVIEW, s1.Suggestion)
	assert.Equal(t, 2, len(s1.Cuts))
	assert.NotNil(t, s2)
	assert.Equal(t, int64(3), s2.OffsetBegin)
	assert.Equal(t, int64(3), s2.OffsetEnd)
	assert.Equal(t, BLOCK, s2.Suggestion)
	assert.Equal(t, 1, len(s2.Cuts))

}

func TestVideoSceneResult(t *testing.T) {
	var v = VideoSceneResult{}

	v = v.Append(SegmentResult{Suggestion: PASS})
	assert.Equal(t, PASS, v.Suggestion)
	assert.Equal(t, 1, len(v.Segments))

	v = v.Append(SegmentResult{Suggestion: PASS})
	assert.Equal(t, PASS, v.Suggestion)
	assert.Equal(t, 2, len(v.Segments))

	v = v.Append(SegmentResult{Suggestion: BLOCK})
	assert.Equal(t, BLOCK, v.Suggestion)
	assert.Equal(t, 3, len(v.Segments))
}

func TestParseCutPulpResult(t *testing.T) {
	cut1 := video.CutResult{Offset: 1, URI: "", Result: censor.PulpResult{Label: 1}}
	bs, err := json.Marshal(cut1)
	assert.NoError(t, err)
	var cut2 OriginCutResult
	err = json.Unmarshal(bs, &cut2)
	assert.NoError(t, err)
	ret := ParseCutPulpResult(cut2, PulpThreshold{})
	assert.Equal(t, int64(1), ret.Offset)
	assert.Equal(t, REVIEW, ret.Suggestion)
}

func TestParseCutTerrorResult(t *testing.T) {
	cut1 := video.CutResult{Offset: 1, URI: "", Result: censor.TerrorResult{Label: 1}}
	bs, err := json.Marshal(cut1)
	assert.NoError(t, err)
	var cut2 OriginCutResult
	err = json.Unmarshal(bs, &cut2)
	assert.NoError(t, err)
	ret := ParseCutTerrorResult(cut2, TerrorThreshold{})
	assert.Equal(t, int64(1), ret.Offset)
	assert.Equal(t, BLOCK, ret.Suggestion)
}

func TestParseCutPoliticianResult(t *testing.T) {
	cut1 := video.CutResult{Offset: 1, URI: "", Result: censor.FaceSearchResult{
		Detections: []censor.FaceSearchDetail{
			func() (ret censor.FaceSearchDetail) {
				ret.Value.Name = "X"
				return
			}(),
		},
	}}
	bs, err := json.Marshal(cut1)
	assert.NoError(t, err)
	var cut2 OriginCutResult
	err = json.Unmarshal(bs, &cut2)
	assert.NoError(t, err)
	ret := ParseCutPoliticianResult(cut2, PoliticianThreshold{})
	assert.Equal(t, int64(1), ret.Offset)
	assert.Equal(t, BLOCK, ret.Suggestion)
}

func TestParseOriginVideoOPResult(t *testing.T) {

	ret1 := video.EndResult{}
	ret1.Result.Segments = []video.SegmentResult{
		video.SegmentResult{Cuts: []video.CutResult{
			{Offset: 1, Result: censor.PulpResult{Label: 1}},
			{Offset: 2, Result: censor.PulpResult{Label: 1}},
			{Offset: 3, Result: censor.PulpResult{Label: 1}},
		}},
		video.SegmentResult{Cuts: []video.CutResult{
			{Offset: 4, Result: censor.PulpResult{Label: 2}},
		}},
		video.SegmentResult{Cuts: []video.CutResult{
			{Offset: 5, Result: censor.PulpResult{Label: 0}},
		}},
	}
	bs, err := json.Marshal(ret1.Result)
	assert.NoError(t, err)
	var ret2 OriginVideoOPResult
	err = json.Unmarshal(bs, &ret2)
	assert.NoError(t, err)

	ret3 := ParseOriginVideoOPResult(ret2,
		func(cut OriginCutResult) CutResult {
			var ret ImagePulpResp
			_ = json.Unmarshal(cut.Result, &ret)
			resp := ParseImagePulpResp(ret, PulpThreshold{})
			return CutResult{Suggestion: resp.Suggestion, Result: resp.Result}
		},
	)
	assert.Equal(t, BLOCK, ret3.Suggestion)
	assert.Equal(t, 3, len(ret3.Segments))
}
