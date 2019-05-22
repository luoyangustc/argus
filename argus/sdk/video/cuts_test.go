package video

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestWithInterval(t *testing.T) {
	f := func(option CutOpOption, offsetMS int64) int64 {
		cs := &cuts{}
		cut := Cut{cuts: cs, is_init: true}
		_ = option(&cut)
		return cs.offsetAdjust(offsetMS)
	}
	{
		option := WithInterval(2000, 1000)
		assert.Equal(t, int64(0), f(option, 0))
		assert.Equal(t, int64(0), f(option, 900))
		assert.Equal(t, int64(2000), f(option, 1000))
		assert.Equal(t, int64(2000), f(option, 1100))
		assert.Equal(t, int64(2000), f(option, 1900))
		assert.Equal(t, int64(2000), f(option, 2000))
		assert.Equal(t, int64(2000), f(option, 2100))
	}
}

func TestCuts(t *testing.T) {

	{
		cuts, _ := CreateCutOP(
			func(context.Context, *Cut) (interface{}, error) {
				return "FOO", nil
			},
		)
		assert.NotNil(t, cuts)

		resps := cuts.Append(context.Background(), CutRequest{OffsetMS: 1})
		assert.Equal(t, 1, len(resps))
		assert.Equal(t, int64(1), resps[0].OffsetMS)
		assert.Equal(t, "FOO", resps[0].Result)

		resps = cuts.Append(context.Background(),
			CutRequest{OffsetMS: 2},
			CutRequest{OffsetMS: 3})
		assert.Equal(t, 2, len(resps))
		assert.Equal(t, int64(2), resps[0].OffsetMS)
		assert.Equal(t, "FOO", resps[0].Result)
		assert.Equal(t, int64(3), resps[1].OffsetMS)
		assert.Equal(t, "FOO", resps[1].Result)

		resps = cuts.End(context.Background())
		assert.Equal(t, 0, len(resps))
	}

	{
		cuts, _ := CreateCutOP(
			func(_ context.Context, cut *Cut) (interface{}, error) {
				return cut.GetRoundResp(0, "")
			},
			WithRoundCutOP(0, "",
				func(context.Context, CutRequest) (interface{}, error) {
					return "FOO", nil
				}),
		)
		assert.NotNil(t, cuts)

		resps := cuts.Append(context.Background(), CutRequest{OffsetMS: 1})
		assert.Equal(t, 1, len(resps))
		assert.Equal(t, int64(1), resps[0].OffsetMS)
		assert.Equal(t, "FOO", resps[0].Result)

		resps = cuts.Append(context.Background(),
			CutRequest{OffsetMS: 2},
			CutRequest{OffsetMS: 3})
		assert.Equal(t, 2, len(resps))
		assert.Equal(t, int64(2), resps[0].OffsetMS)
		assert.Equal(t, "FOO", resps[0].Result)
		assert.Equal(t, int64(3), resps[1].OffsetMS)
		assert.Equal(t, "FOO", resps[1].Result)

		resps = cuts.End(context.Background())
		assert.Equal(t, 0, len(resps))
	}

	{
		var fA = func(context.Context, CutRequest) (interface{}, error) {
			return 10, nil
		}
		var fB = func(context.Context, CutRequest) (interface{}, error) {
			return 100, nil
		}
		cuts, _ := CreateCutOP(
			func(_ context.Context, cut *Cut) (interface{}, error) {
				var sum int = 0
				if resp, _ := cut.GetRoundResp(0, "A"); resp != nil {
					sum += resp.(int)
				}
				if resp, _ := cut.GetRoundResp(-500, "A"); resp != nil {
					sum += resp.(int)
				}
				if resp, _ := cut.GetRoundResp(500, "A"); resp != nil {
					sum += resp.(int)
				}
				if resp, _ := cut.GetRoundResp(-500, "B"); resp != nil {
					sum += resp.(int)
				}
				if resp, _ := cut.GetRoundResp(500, "B"); resp != nil {
					sum += resp.(int)
				}
				return sum, nil
			},
			WithCutFilter(func(offsetMS int64) bool { return offsetMS%2000 == 0 }),
			WithRoundCutOP(0, "A", fA),
			WithRoundCutOP(-500, "A", fA),
			WithRoundCutOP(500, "A", fA),
			WithRoundCutOP(-500, "B", fB),
			WithRoundCutOP(500, "B", fB),
		)
		assert.NotNil(t, cuts)

		resps := cuts.Append(context.Background(), CutRequest{OffsetMS: 0})
		assert.Equal(t, 0, len(resps))
		resps = cuts.Append(context.Background(), CutRequest{OffsetMS: 500})
		assert.Equal(t, 1, len(resps))
		assert.Equal(t, int64(0), resps[0].OffsetMS)
		assert.Equal(t, 120, resps[0].Result)
		resps = cuts.Append(context.Background(), CutRequest{OffsetMS: 1000})
		assert.Equal(t, 0, len(resps))
		resps = cuts.Append(context.Background(), CutRequest{OffsetMS: 1500})
		assert.Equal(t, 0, len(resps))
		resps = cuts.Append(context.Background(), CutRequest{OffsetMS: 2000})
		assert.Equal(t, 0, len(resps))
		resps = cuts.Append(context.Background(), CutRequest{OffsetMS: 2500})
		assert.Equal(t, 1, len(resps))
		assert.Equal(t, int64(2000), resps[0].OffsetMS)
		assert.Equal(t, 230, resps[0].Result)
		resps = cuts.Append(context.Background(), CutRequest{OffsetMS: 3000})
		assert.Equal(t, 0, len(resps))
		resps = cuts.Append(context.Background(), CutRequest{OffsetMS: 3500})
		assert.Equal(t, 0, len(resps))
		resps = cuts.Append(context.Background(), CutRequest{OffsetMS: 4000})
		assert.Equal(t, 0, len(resps))

		resps = cuts.End(context.Background())
		assert.Equal(t, 1, len(resps))
		assert.Equal(t, int64(4000), resps[0].OffsetMS)
		assert.Equal(t, 120, resps[0].Result)
	}

}

func TestMultiCuts(t *testing.T) {

	cuts1, _ := CreateCutOP(
		func(context.Context, *Cut) (interface{}, error) {
			return "FOO", nil
		},
		WithCutFilter(func(offsetMS int64) bool { return offsetMS%2000 == 0 }),
	)
	var fA = func(context.Context, CutRequest) (interface{}, error) {
		return 10, nil
	}
	var fB = func(context.Context, CutRequest) (interface{}, error) {
		return 100, nil
	}
	cuts2, _ := CreateCutOP(
		func(_ context.Context, cut *Cut) (interface{}, error) {
			var sum int = 0
			if resp, _ := cut.GetRoundResp(0, "A"); resp != nil {
				sum += resp.(int)
			}
			if resp, _ := cut.GetRoundResp(-500, "A"); resp != nil {
				sum += resp.(int)
			}
			if resp, _ := cut.GetRoundResp(500, "A"); resp != nil {
				sum += resp.(int)
			}
			if resp, _ := cut.GetRoundResp(-500, "B"); resp != nil {
				sum += resp.(int)
			}
			if resp, _ := cut.GetRoundResp(500, "B"); resp != nil {
				sum += resp.(int)
			}
			return sum, nil
		},
		WithCutFilter(func(offsetMS int64) bool { return offsetMS%2000 == 0 }),
		WithRoundCutOP(0, "A", fA),
		WithRoundCutOP(-500, "A", fA),
		WithRoundCutOP(500, "A", fA),
		WithRoundCutOP(-500, "B", fB),
		WithRoundCutOP(500, "B", fB),
	)
	cuts := CreateMultiCutsPipe(map[string]CutsPipe{"1": cuts1, "2": cuts2})

	resps := cuts.Append(context.Background(), CutRequest{OffsetMS: 0})
	assert.Equal(t, 0, len(resps))
	resps = cuts.Append(context.Background(), CutRequest{OffsetMS: 500})
	assert.Equal(t, 1, len(resps))
	assert.Equal(t, int64(0), resps[0].OffsetMS)
	assert.Equal(t, 2, len(resps[0].Result.(map[string]CutResponse)))
	assert.Equal(t, "FOO", resps[0].Result.(map[string]CutResponse)["1"].Result)
	assert.Equal(t, 120, resps[0].Result.(map[string]CutResponse)["2"].Result)
	resps = cuts.Append(context.Background(), CutRequest{OffsetMS: 1000})
	assert.Equal(t, 0, len(resps))
	resps = cuts.Append(context.Background(), CutRequest{OffsetMS: 1500})
	assert.Equal(t, 0, len(resps))
	resps = cuts.Append(context.Background(), CutRequest{OffsetMS: 2000})
	assert.Equal(t, 0, len(resps))
	resps = cuts.Append(context.Background(), CutRequest{OffsetMS: 2500})
	assert.Equal(t, 1, len(resps))
	assert.Equal(t, int64(2000), resps[0].OffsetMS)
	assert.Equal(t, 2, len(resps[0].Result.(map[string]CutResponse)))
	assert.Equal(t, "FOO", resps[0].Result.(map[string]CutResponse)["1"].Result)
	assert.Equal(t, 230, resps[0].Result.(map[string]CutResponse)["2"].Result)
	resps = cuts.Append(context.Background(), CutRequest{OffsetMS: 3000})
	assert.Equal(t, 0, len(resps))
	resps = cuts.Append(context.Background(), CutRequest{OffsetMS: 3500})
	assert.Equal(t, 0, len(resps))
	resps = cuts.Append(context.Background(), CutRequest{OffsetMS: 4000})
	assert.Equal(t, 0, len(resps))

	resps = cuts.End(context.Background())
	assert.Equal(t, 1, len(resps))
	assert.Equal(t, int64(4000), resps[0].OffsetMS)
	assert.Equal(t, 2, len(resps[0].Result.(map[string]CutResponse)))
	assert.Equal(t, "FOO", resps[0].Result.(map[string]CutResponse)["1"].Result)
	assert.Equal(t, 120, resps[0].Result.(map[string]CutResponse)["2"].Result)

}
