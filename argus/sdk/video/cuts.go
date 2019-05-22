package video

import (
	"container/list"
	"context"
	"fmt"
	"math"
	"sync"

	"qiniu.com/argus/com/util"
)

type CutRequest struct {
	OffsetMS int64
	Body     []byte
}

type CutResponse struct {
	OffsetMS int64
	Result   interface{}
	Error    error
}

type CutsPipe interface {
	Append(context.Context, ...CutRequest) []CutResponse
	End(context.Context) []CutResponse
}

////////////////////////////////////////////////////////////////////////////////

type CutOP func(context.Context, CutRequest) (interface{}, error)
type CutOpOption func(cut *Cut) error

func WithRoundCutOP(intervalMS int64, tag string, cutOP CutOP) CutOpOption {
	return func(cut *Cut) error {
		if cut.is_init {
			cut.roundOPs[cut.roundKey(intervalMS, tag)] =
				roundOP{IntervalMS: intervalMS, Tag: tag, CutOP: cutOP}
		}
		return nil
	}
}

func WithOffsetAdjust(f func(int64) int64) CutOpOption {
	return func(cut *Cut) error {
		if cut.is_init {
			cut.offsetAdjust = f
		}
		return nil
	}
}

func WithInterval(intervalMS int64, rangeMS int64) CutOpOption {
	return WithOffsetAdjust(func(offsetMS int64) int64 {
		d := offsetMS % intervalMS
		if d < rangeMS {
			return offsetMS - d
		} else if (intervalMS - d) <= rangeMS {
			return offsetMS + intervalMS - d
		} else {
			return offsetMS
		}
	})
}

func WithCutFilter(f func(int64) bool) CutOpOption {
	return func(cut *Cut) error {
		if !cut.is_init {
			cut.is_target = f(cut.offsetMS)
		}
		return nil
	}
}

type roundOP struct {
	IntervalMS int64
	Tag        string
	CutOP
}
type Cut struct {
	*cuts
	is_init bool

	offsetMS       int64
	originOffsetMS int64
	is_target      bool
}

func (c *Cut) Body() ([]byte, error) {
	if req, ok := c.cuts.cache_cuts[c.offsetMS]; ok {
		return req.Body, nil
	}
	return nil, nil
}
func (c *Cut) GetRoundResp(intervalMS int64, tag string) (interface{}, error) {
	resp, ok := c.cuts.cache_rounds.Load(c.cuts.roundKey(c.offsetMS+intervalMS, tag))
	if !ok {
		return nil, nil
	}
	return resp.(roundResp).Resp, resp.(roundResp).Error
}

type roundResp struct {
	OffsetMS int64
	Tag      string
	Resp     interface{}
	Error    error
}

var _ CutsPipe = &cuts{}

type cuts struct {
	op           func(context.Context, *Cut) (interface{}, error)
	opOptions    []CutOpOption
	roundOPs     map[string]roundOP
	offsetAdjust func(int64) int64

	cache_targets *list.List // []*Cut
	cache_cuts    map[int64]CutRequest
	cache_rounds  *sync.Map // map[string]roundResp
}

func (cs cuts) roundKey(intervalMS int64, tag string) string {
	return fmt.Sprintf("%d-%s", intervalMS, tag)
}

func (cs *cuts) cacheClean(offsetMS int64) {
	var first = offsetMS
	for _, ro := range cs.roundOPs {
		if offsetMS+ro.IntervalMS < first {
			first = offsetMS + ro.IntervalMS
		}
	}
	var overdueOffsets = make([]int64, 0, len(cs.cache_cuts))
	for offsetMS, _ := range cs.cache_cuts {
		if offsetMS <= first {
			overdueOffsets = append(overdueOffsets, offsetMS)
		}
	}
	for _, offsetMS := range overdueOffsets {
		delete(cs.cache_cuts, offsetMS)
	}
	var overdueKeys = make([]string, 0)
	cs.cache_rounds.Range(func(key, value interface{}) bool {
		if value.(roundResp).OffsetMS <= first {
			overdueKeys = append(overdueKeys, key.(string))
		}
		return true
	})
	for _, key := range overdueKeys {
		cs.cache_rounds.Delete(key)
	}
}

func (cs *cuts) runRound(
	ctx context.Context,
	g *sync.WaitGroup, index *int,
	offsetMS int64) bool {
	var done bool = true
	for _, round := range cs.roundOPs {
		var offset = round.IntervalMS + offsetMS
		var roundKey = cs.roundKey(offset, round.Tag)
		if _, ok := cs.cache_rounds.Load(roundKey); ok {
			continue
		}
		req, ok := cs.cache_cuts[offset]
		if !ok {
			if round.IntervalMS > 0 {
				done = false
			}
			continue
		}
		*index += 1
		g.Add(1)
		go func(ctx context.Context, op CutOP, tag string) {
			defer g.Done()
			resp, err := op(ctx, req)
			cs.cache_rounds.Store(roundKey,
				roundResp{
					OffsetMS: req.OffsetMS,
					Tag:      tag,
					Resp:     resp,
					Error:    err,
				})
		}(util.SpawnContext2(ctx, *index), round.CutOP, round.Tag)
	}
	return done
}

func (cs *cuts) Append(ctx context.Context, reqs ...CutRequest) []CutResponse {
	for _, req := range reqs {
		var cut = &Cut{
			cuts:           cs,
			originOffsetMS: req.OffsetMS,
			offsetMS:       cs.offsetAdjust(req.OffsetMS),
			is_target:      true,
		}
		for _, option := range cs.opOptions {
			// 不中断处理过程
			// if err := option(cut); err != nil {
			// 	return nil, err
			// }
			_ = option(cut)
		}
		cs.cache_cuts[cut.offsetMS] = req
		if cut.is_target {
			cs.cache_targets.PushBack(cut)
		}
	}

	var g = sync.WaitGroup{}
	var index = 0
	var cuts_done = make([]*list.Element, 0, cs.cache_targets.Len())
	for e := cs.cache_targets.Front(); e != nil; e = e.Next() {
		if cs.runRound(ctx, &g, &index, e.Value.(*Cut).offsetMS) {
			var ee = e
			cuts_done = append(cuts_done, ee)
		}
	}
	g.Wait()

	var (
		rets = make([]CutResponse, len(cuts_done))
		wg   sync.WaitGroup
	)

	for i := 0; i < len(cuts_done); i++ {
		wg.Add(1)
		go func(ctx context.Context, index int) {
			defer wg.Done()
			cut := cuts_done[index].Value.(*Cut)
			resp, err := cs.op(ctx, cut)
			rets[index] = CutResponse{OffsetMS: cut.originOffsetMS, Result: resp, Error: err}
		}(util.SpawnContext2(ctx, i), i)
	}
	wg.Wait()

	for _, e := range cuts_done {
		cs.cache_targets.Remove(e)
	}

	if len(rets) > 0 {
		cs.cacheClean(rets[len(rets)-1].OffsetMS)
	}
	return rets
}

func (cs *cuts) End(ctx context.Context) []CutResponse {
	var rets = make([]CutResponse, 0, cs.cache_targets.Len())
	for e := cs.cache_targets.Front(); e != nil; e = e.Next() {
		var cut = e.Value.(*Cut)
		resp, err := cs.op(ctx, cut)
		rets = append(rets, CutResponse{OffsetMS: cut.originOffsetMS, Result: resp, Error: err})
	}
	return rets
}

func CreateCutOP(
	op func(context.Context, *Cut) (interface{}, error),
	option ...CutOpOption) (CutsPipe, error) {
	cs := &cuts{
		op:            op,
		opOptions:     option,
		roundOPs:      make(map[string]roundOP),
		offsetAdjust:  func(offsetMS int64) int64 { return offsetMS },
		cache_targets: list.New(),
		cache_cuts:    make(map[int64]CutRequest),
		cache_rounds:  new(sync.Map),
	}

	cut := &Cut{cuts: cs, is_init: true}
	for _, option_ := range option {
		if err := option_(cut); err != nil {
			return nil, err
		}
	}

	return cs, nil
}

////////////////////////////////////////////////////////////////////////////////

type multiCutsPipe struct {
	resps     *list.List // []CutResponse
	offsetMSs map[string]int64
	pipes     map[string]CutsPipe
}

func CreateMultiCutsPipe(pipes map[string]CutsPipe) CutsPipe {
	offsetMSs := make(map[string]int64)
	for name, _ := range pipes {
		offsetMSs[name] = -1
	}
	return &multiCutsPipe{
		resps:     list.New(),
		offsetMSs: offsetMSs,
		pipes:     pipes,
	}
}

func (mp *multiCutsPipe) run(ctx context.Context, f func(CutsPipe) []CutResponse) []CutResponse {
	var (
		wg    sync.WaitGroup
		index = -1
		resps = make([]struct {
			Name  string
			Resps []CutResponse
		}, len(mp.pipes))
	)
	for name, pipe := range mp.pipes {
		wg.Add(1)
		index++
		go func(i int, name string, pipe CutsPipe) {
			defer wg.Done()
			resps[i].Resps = f(pipe)
			resps[i].Name = name
		}(index, name, pipe)
	}
	wg.Wait()

	for _, resp_ := range resps {
		name := resp_.Name
		for _, resp := range resp_.Resps {
			var e *list.Element
			var resp0 = CutResponse{OffsetMS: resp.OffsetMS, Result: make(map[string]CutResponse)}
			for e = mp.resps.Front(); e != nil; e = e.Next() {
				if e.Value.(CutResponse).OffsetMS == resp.OffsetMS {
					resp0 = e.Value.(CutResponse)
					break
				}
			}
			if e == nil {
				mp.resps.PushBack(resp0)
			}
			resp0.Result.(map[string]CutResponse)[name] = resp
			mp.offsetMSs[name] = resp.OffsetMS
		}
	}

	var first int64 = math.MaxInt64
	for _, offsetMS := range mp.offsetMSs {
		if offsetMS < first {
			first = offsetMS
		}
	}
	rets := make([]CutResponse, 0, mp.resps.Len())
	for e := mp.resps.Front(); e != nil; {
		if e.Value.(CutResponse).OffsetMS <= first {
			e0 := e
			e = e.Next()
			rets = append(rets, mp.resps.Remove(e0).(CutResponse))
		} else {
			e = e.Next()
		}
	}
	return rets
}

func (mp *multiCutsPipe) Append(ctx context.Context, req ...CutRequest) []CutResponse {
	return mp.run(ctx, func(pipe CutsPipe) []CutResponse { return pipe.Append(ctx, req...) })
}

func (mp *multiCutsPipe) End(ctx context.Context) []CutResponse {
	return mp.run(ctx, func(pipe CutsPipe) []CutResponse { return pipe.End(ctx) })
}
