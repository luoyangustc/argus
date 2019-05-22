package video

import (
	"container/list"
	"context"
	"testing"

	"qiniu.com/argus/video/vframe"

	"github.com/stretchr/testify/assert"
)

func TestFixedTimeline(t *testing.T) {
	{
		tl := newFixedTimeline(500, 2000, []int64{-500, 0, 500})

		var (
			/*
				skip     bool
				misses   []struct {
					Offset int64
					Select byte
				}
			*/
			isSelect byte
			tails    []struct {
				Offset int64
				Select byte
			}
		)

		_, _, isSelect = tl.Update(0)
		assert.Equal(t, __FIXED_SELECT, isSelect&__FIXED_SELECT)
		assert.Equal(t, __FIXED_TARGET, isSelect&__FIXED_TARGET)
		assert.Equal(t, __FIXED_ZERO, isSelect&__FIXED_ROUND)

		_, _, isSelect = tl.Update(500)
		assert.Equal(t, __FIXED_SELECT, isSelect&__FIXED_SELECT)
		assert.Equal(t, __FIXED_ZERO, isSelect&__FIXED_TARGET)
		assert.Equal(t, __FIXED_ROUND, isSelect&__FIXED_ROUND)

		_, _, isSelect = tl.Update(1000)
		assert.Equal(t, __FIXED_ZERO, isSelect&__FIXED_SELECT)
		assert.Equal(t, __FIXED_ZERO, isSelect&__FIXED_TARGET)
		assert.Equal(t, __FIXED_ZERO, isSelect&__FIXED_ROUND)

		_, _, isSelect = tl.Update(1500)
		assert.Equal(t, __FIXED_SELECT, isSelect&__FIXED_SELECT)
		assert.Equal(t, __FIXED_ZERO, isSelect&__FIXED_TARGET)
		assert.Equal(t, __FIXED_ZERO, isSelect&__FIXED_ROUND)

		_, _, isSelect = tl.Update(2000)
		assert.Equal(t, __FIXED_SELECT, isSelect&__FIXED_SELECT)
		assert.Equal(t, __FIXED_TARGET, isSelect&__FIXED_TARGET)
		assert.Equal(t, __FIXED_ZERO, isSelect&__FIXED_ROUND)

		_, _, isSelect = tl.Update(2500)
		assert.Equal(t, __FIXED_SELECT, isSelect&__FIXED_SELECT)
		assert.Equal(t, __FIXED_ZERO, isSelect&__FIXED_TARGET)
		assert.Equal(t, __FIXED_ROUND, isSelect&__FIXED_ROUND)

		_, _, isSelect = tl.Update(3000)
		assert.Equal(t, __FIXED_ZERO, isSelect&__FIXED_SELECT)
		assert.Equal(t, __FIXED_ZERO, isSelect&__FIXED_TARGET)
		assert.Equal(t, __FIXED_ZERO, isSelect&__FIXED_ROUND)

		_, _, isSelect = tl.Update(3500)
		assert.Equal(t, __FIXED_SELECT, isSelect&__FIXED_SELECT)
		assert.Equal(t, __FIXED_ZERO, isSelect&__FIXED_TARGET)
		assert.Equal(t, __FIXED_ZERO, isSelect&__FIXED_ROUND)

		_, _, isSelect = tl.Update(4000)
		assert.Equal(t, __FIXED_SELECT, isSelect&__FIXED_SELECT)
		assert.Equal(t, __FIXED_TARGET, isSelect&__FIXED_TARGET)
		assert.Equal(t, __FIXED_ZERO, isSelect&__FIXED_ROUND)

		_, _, isSelect = tl.Update(4500)
		assert.Equal(t, __FIXED_SELECT, isSelect&__FIXED_SELECT)
		assert.Equal(t, __FIXED_ZERO, isSelect&__FIXED_TARGET)
		assert.Equal(t, __FIXED_ROUND, isSelect&__FIXED_ROUND)

		_, _, isSelect = tl.Update(5000)
		assert.Equal(t, __FIXED_ZERO, isSelect&__FIXED_SELECT)
		assert.Equal(t, __FIXED_ZERO, isSelect&__FIXED_TARGET)
		assert.Equal(t, __FIXED_ZERO, isSelect&__FIXED_ROUND)

		_, _, isSelect = tl.Update(5500)
		assert.Equal(t, __FIXED_SELECT, isSelect&__FIXED_SELECT)
		assert.Equal(t, __FIXED_ZERO, isSelect&__FIXED_TARGET)
		assert.Equal(t, __FIXED_ZERO, isSelect&__FIXED_ROUND)

		tails = tl.Tail() // creates one mock cut, there's no __FIXED_ROUND cut mocked.
		assert.Equal(t, 1, len(tails))
		assert.Equal(t, int64(6000), tails[0].Offset)
		assert.Equal(t, __FIXED_SELECT, tails[0].Select&__FIXED_SELECT)
		assert.Equal(t, __FIXED_TARGET, tails[0].Select&__FIXED_TARGET)
		assert.Equal(t, __FIXED_ZERO, tails[0].Select&__FIXED_ROUND)
	}

	{
		tl := newFixedTimeline(100, 400, []int64{-500, 0, 500})

		var (
			/*
				skip     bool
				misses   []struct {
					Offset int64
					Select byte
				}
			*/
			isSelect byte
			tails    []struct {
				Offset int64
				Select byte
			}
		)

		_, _, isSelect = tl.Update(0)
		assert.Equal(t, __FIXED_SELECT, isSelect&__FIXED_SELECT)
		assert.Equal(t, __FIXED_TARGET, isSelect&__FIXED_TARGET)
		assert.Equal(t, __FIXED_ZERO, isSelect&__FIXED_ROUND)

		_, _, isSelect = tl.Update(100)
		assert.Equal(t, __FIXED_SELECT, isSelect&__FIXED_SELECT)
		assert.Equal(t, __FIXED_ZERO, isSelect&__FIXED_TARGET)
		assert.Equal(t, __FIXED_ROUND, isSelect&__FIXED_ROUND)

		_, _, isSelect = tl.Update(200)
		assert.Equal(t, __FIXED_ZERO, isSelect&__FIXED_SELECT)
		assert.Equal(t, __FIXED_ZERO, isSelect&__FIXED_TARGET)
		assert.Equal(t, __FIXED_ZERO, isSelect&__FIXED_ROUND)

		_, _, isSelect = tl.Update(300)
		assert.Equal(t, __FIXED_SELECT, isSelect&__FIXED_SELECT)
		assert.Equal(t, __FIXED_ZERO, isSelect&__FIXED_TARGET)
		assert.Equal(t, __FIXED_ZERO, isSelect&__FIXED_ROUND)

		_, _, isSelect = tl.Update(400)
		assert.Equal(t, __FIXED_SELECT, isSelect&__FIXED_SELECT)
		assert.Equal(t, __FIXED_TARGET, isSelect&__FIXED_TARGET)
		assert.Equal(t, __FIXED_ZERO, isSelect&__FIXED_ROUND)

		_, _, isSelect = tl.Update(500)
		assert.Equal(t, __FIXED_SELECT, isSelect&__FIXED_SELECT)
		assert.Equal(t, __FIXED_ZERO, isSelect&__FIXED_TARGET)
		assert.Equal(t, __FIXED_ROUND, isSelect&__FIXED_ROUND)

		_, _, isSelect = tl.Update(600)
		assert.Equal(t, __FIXED_ZERO, isSelect&__FIXED_SELECT)
		assert.Equal(t, __FIXED_ZERO, isSelect&__FIXED_TARGET)
		assert.Equal(t, __FIXED_ZERO, isSelect&__FIXED_ROUND)

		_, _, isSelect = tl.Update(700)
		assert.Equal(t, __FIXED_SELECT, isSelect&__FIXED_SELECT)
		assert.Equal(t, __FIXED_ZERO, isSelect&__FIXED_TARGET)
		assert.Equal(t, __FIXED_ZERO, isSelect&__FIXED_ROUND)

		_, _, isSelect = tl.Update(800)
		assert.Equal(t, __FIXED_SELECT, isSelect&__FIXED_SELECT)
		assert.Equal(t, __FIXED_TARGET, isSelect&__FIXED_TARGET)
		assert.Equal(t, __FIXED_ZERO, isSelect&__FIXED_ROUND)

		tails = tl.Tail() // mocked the __FIXED_ROUND cut
		assert.Equal(t, 5, len(tails))
		assert.Equal(t, int64(900), tails[0].Offset)
		assert.Equal(t, __FIXED_SELECT, tails[0].Select&__FIXED_SELECT)
		assert.Equal(t, __FIXED_ZERO, tails[0].Select&__FIXED_TARGET)
		assert.Equal(t, __FIXED_ROUND, tails[0].Select&__FIXED_ROUND)
	}

}

type _MockVframeJob struct {
	ch chan vframe.CutResponse
}

func (mock _MockVframeJob) Cuts() <-chan vframe.CutResponse { return mock.ch }
func (mock _MockVframeJob) Error() error                    { return nil }
func (mock _MockVframeJob) Stop()                           { close(mock.ch) }

func TestFixedCuts(t *testing.T) {

	var (
		match []interface{}

		newCut = func(offset int64) vframe.CutResponse {
			resp := vframe.CutResponse{}
			resp.Result.Cut.Offset = offset
			return resp
		}
	)

	{
		var (
			job  = _MockVframeJob{ch: make(chan vframe.CutResponse, 20)}
			cuts = &simpleCuts{
				CutsV2: newFixedAggCutsV2(500, 2000, []int64{-500, 0, 500}),
				cache:  list.New(),
			}
			current = list.New()
		)
		{
			assert.Equal(t, 4, len(cuts.CutsV2.(*fixedAggCutsV2).cacheSelect))
			assert.Equal(t, -1, cuts.CutsV2.(*fixedAggCutsV2).cacheSelect[0])
			assert.Equal(t, -2, cuts.CutsV2.(*fixedAggCutsV2).cacheSelect[1])
			assert.Equal(t, -1, cuts.CutsV2.(*fixedAggCutsV2).cacheSelect[2])
			assert.Equal(t, 0, cuts.CutsV2.(*fixedAggCutsV2).cacheSelect[3])
		}

		cuts.CutsV2.(*fixedAggCutsV2).evalFunc = func(ctx context.Context, uris []string, free bool) ([]interface{}, error) {
			return current.Remove(current.Front()).([]interface{}), nil
		}
		cuts.CutsV2.(*fixedAggCutsV2).aggFunc = func(ctx context.Context, results []interface{}) (interface{}, error) {
			assert.Equal(t, match, results)
			return nil, nil
		}
		cuts.CutsV2.(*fixedAggCutsV2).parseFunc = func(ctx context.Context, result interface{}) ([]string, []float32, error) {
			return nil, nil, nil
		}
		cuts.Job = job

		{
			job.ch <- newCut(0)
			current.PushBack([]interface{}{0})
			job.ch <- newCut(500)
			current.PushBack([]interface{}{500})

			match = []interface{}{nil, 0, 500}

			ret, ok := cuts.Next(context.Background())
			assert.True(t, ok)
			assert.Equal(t, int64(0), ret.Offset)
		}
		{
			job.ch <- newCut(1000)
			job.ch <- newCut(1500)
			current.PushBack([]interface{}{1500})
			job.ch <- newCut(2000)
			current.PushBack([]interface{}{2000})
			job.ch <- newCut(2500)
			current.PushBack([]interface{}{2500})

			match = []interface{}{1500, 2000, 2500}

			ret, ok := cuts.Next(context.Background())
			assert.True(t, ok)
			assert.Equal(t, int64(2000), ret.Offset)
		}
		{
			job.ch <- newCut(3000)
			job.ch <- newCut(3500)
			current.PushBack([]interface{}{3500})
			job.ch <- newCut(4000)
			current.PushBack([]interface{}{4000})
			job.Stop()

			match = []interface{}{3500, 4000, nil} // there's one mock cut

			ret, ok := cuts.Next(context.Background())
			assert.True(t, ok)
			assert.Equal(t, int64(4000), ret.Offset)

			_, ok = cuts.Next(context.Background())
			assert.False(t, ok)
		}
	}

	{
		var (
			job  = _MockVframeJob{ch: make(chan vframe.CutResponse, 20)}
			cuts = &simpleCuts{
				CutsV2: newFixedAggCutsV2(100, 400, []int64{-500, 0, 500}),
				cache:  list.New(),
			}
			current = list.New()
		)
		{
			assert.Equal(t, 4, len(cuts.CutsV2.(*fixedAggCutsV2).cacheSelect))
			assert.Equal(t, -5, cuts.CutsV2.(*fixedAggCutsV2).cacheSelect[0])
			assert.Equal(t, -10, cuts.CutsV2.(*fixedAggCutsV2).cacheSelect[1])
			assert.Equal(t, -5, cuts.CutsV2.(*fixedAggCutsV2).cacheSelect[2])
			assert.Equal(t, 0, cuts.CutsV2.(*fixedAggCutsV2).cacheSelect[3])
		}

		cuts.CutsV2.(*fixedAggCutsV2).evalFunc = func(ctx context.Context, uris []string, free bool) ([]interface{}, error) {
			return current.Remove(current.Front()).([]interface{}), nil
		}
		cuts.CutsV2.(*fixedAggCutsV2).aggFunc = func(ctx context.Context, results []interface{}) (interface{}, error) {
			assert.Equal(t, match, results)
			return nil, nil
		}
		cuts.CutsV2.(*fixedAggCutsV2).parseFunc = func(ctx context.Context, result interface{}) ([]string, []float32, error) {
			return nil, nil, nil
		}
		cuts.Job = job

		{
			job.ch <- newCut(0)
			current.PushBack([]interface{}{0})
			job.ch <- newCut(100)
			current.PushBack([]interface{}{100})
			job.ch <- newCut(200)
			job.ch <- newCut(300)
			current.PushBack([]interface{}{300})
			job.ch <- newCut(400)
			current.PushBack([]interface{}{400})
			job.ch <- newCut(500)
			current.PushBack([]interface{}{500})

			match = []interface{}{nil, 0, 500}

			ret, ok := cuts.Next(context.Background())
			assert.True(t, ok)
			assert.Equal(t, int64(0), ret.Offset)
		}
		{
			job.ch <- newCut(600)
			job.ch <- newCut(700)
			current.PushBack([]interface{}{700})
			job.ch <- newCut(800)
			current.PushBack([]interface{}{800})
			job.ch <- newCut(900)
			current.PushBack([]interface{}{900})

			match = []interface{}{nil, 400, 900}

			ret, ok := cuts.Next(context.Background())
			assert.True(t, ok)
			assert.Equal(t, int64(400), ret.Offset)
		}
		{
			job.Stop()
		}
	}

}
