package video

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestCutSelect(t *testing.T) {

	{
		sel := selectCut([]struct {
			Name   string  `json:"label"`
			Select int     `json:"select"`
			Score  float32 `json:"score"`
		}{}, true)

		ok, ret := sel(CutResultWithLabels{Labels: []ResultLabel{{Name: "A"}}})
		assert.True(t, ok)
		assert.Equal(t, 1, len(ret.Labels))
		assert.Equal(t, "A", ret.Labels[0].Name)
	}
	{
		sel := selectCut([]struct {
			Name   string  `json:"label"`
			Select int     `json:"select"`
			Score  float32 `json:"score"`
		}{{Name: "A", Select: LABEL_SELECT_CHOOSE_ONLY, Score: 0.5}}, true)

		ok, ret := sel(CutResultWithLabels{Labels: []ResultLabel{{Name: "B"}}})
		assert.False(t, ok)

		ok, ret = sel(CutResultWithLabels{Labels: []ResultLabel{{Name: "A"}, {Name: "B"}}})
		assert.False(t, ok)

		ok, ret = sel(CutResultWithLabels{Labels: []ResultLabel{{Name: "A", Score: 0.6}, {Name: "B"}}})
		assert.True(t, ok)
		assert.Equal(t, 1, len(ret.Labels))
		assert.Equal(t, "A", ret.Labels[0].Name)
	}
	{
		sel := selectCut([]struct {
			Name   string  `json:"label"`
			Select int     `json:"select"`
			Score  float32 `json:"score"`
		}{{Name: "A", Select: LABEL_SELECT_INGORE, Score: 0.5}}, false)

		ok, ret := sel(CutResultWithLabels{Labels: []ResultLabel{{Name: "A"}}})
		assert.False(t, ok)

		ok, ret = sel(CutResultWithLabels{Labels: []ResultLabel{{Name: "B"}}})
		assert.True(t, ok)
		assert.Equal(t, 1, len(ret.Labels))
		assert.Equal(t, "B", ret.Labels[0].Name)

		ok, ret = sel(CutResultWithLabels{Labels: []ResultLabel{{Name: "A"}, {Name: "B"}}})
		assert.True(t, ok)
		assert.Equal(t, 1, len(ret.Labels))
		assert.Equal(t, "B", ret.Labels[0].Name)
	}
}

func TestMergeLabels(t *testing.T) {
	labels := mergeLabels(
		[]ResultLabel{},
		[]ResultLabel{{Name: "A"}}...,
	)
	assert.Equal(t, 1, len(labels))
	assert.Equal(t, "A", labels[0].Name)

	labels = mergeLabels(
		[]ResultLabel{{Name: "A"}, {Name: "B", Score: 0.8}},
		[]ResultLabel{{Name: "A", Score: 0.1}, {Name: "B", Score: 0.9}, {Name: "C", Score: 0.6}}...,
	)
	assert.Equal(t, 3, len(labels))
	assert.Equal(t, "A", labels[0].Name)
	assert.Equal(t, "B", labels[1].Name)
	assert.Equal(t, "C", labels[2].Name)
}

func TestAutoSegmentMeta(t *testing.T) {

	{
		var meta SegmentMeta = &_AutoSegmentMeta{}
		meta.Append(CutResultWithLabels{Labels: []ResultLabel{{Name: "A"}}})
		nw, sr := meta.Next()
		assert.NotNil(t, nw)
		assert.Nil(t, sr)

		meta = nw
		meta.Append(CutResultWithLabels{Labels: []ResultLabel{{Name: "A"}}})
		nw, sr = meta.Next()
		assert.Nil(t, nw)

		meta.Append(CutResultWithLabels{Labels: []ResultLabel{{Name: "B"}}})
		nw, sr = meta.Next()
		assert.NotNil(t, nw)
		assert.Equal(t, 1, len(sr.Labels))
		assert.Equal(t, "A", sr.Labels[0].Name)
	}
}
func TestStaticSegmentMeta(t *testing.T) {

	{
		var meta SegmentMeta = &_StaticSegmentMeta{Interval: 5}
		meta.Append(CutResultWithLabels{CutResult: CutResult{Offset: 1}, Labels: []ResultLabel{{Name: "A"}}})
		nw, sr := meta.Next()
		assert.NotNil(t, nw)
		assert.Nil(t, sr)

		meta = nw
		meta.Append(CutResultWithLabels{CutResult: CutResult{Offset: 2}, Labels: []ResultLabel{{Name: "A"}}})
		nw, sr = meta.Next()
		assert.Nil(t, nw)

		meta.Append(CutResultWithLabels{CutResult: CutResult{Offset: 6}, Labels: []ResultLabel{{Name: "B"}}})
		nw, sr = meta.Next()
		assert.NotNil(t, nw)
		assert.Equal(t, 1, len(sr.Labels))
		assert.Equal(t, "A", sr.Labels[0].Name)
	}
}

func TestSimpleCutOPMeta(t *testing.T) {
	ctx := context.Background()
	opParams := OPParams{}
	opParams.Terminate.Mode = TERMINATE_MODE_SEGMENT
	meta := NewSimpleCutOPMeta(SegmentParams{Mode: 0, Interval: 0}, opParams, nil, nil, true)
	selected, cut := meta.CutSelect(CutResultWithLabels{
		CutResult: CutResult{Offset: 1, URI: "xx", Result: "result"},
		Labels: []ResultLabel{
			ResultLabel{
				Name:  "xx",
				Score: 11,
			},
		},
	})
	if selected {
		meta.Append(ctx, cut)
	}
	// should not panic
}

func TestCutsMeta(t *testing.T) {

	assert.True(t, true)
}
