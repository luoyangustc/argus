package feature_group

import (
	"context"
	"sort"

	"github.com/pkg/errors"
	xlog "github.com/qiniu/xlog.v1"

	"qiniu.com/argus/feature_group/distance"
)

type Memory struct {
	cached *cached
	bs     []byte
}

func NewMemory(config StorageConfig, fetch func(context.Context, Key) ([]byte, error)) *Memory {
	bs := make([]byte, config.Size)
	return &Memory{
		cached: newCached(config, fetch,
			func(ctx context.Context, offset uint64, data []byte) error {
				copy(bs[offset:], data)
				return nil
			},
		),
		bs: bs,
	}
}

type SearchResult struct {
	Items []SearchResultItem
}

type SearchResultItem struct {
	Version HubVersion
	Index   int
	Score   float32
}

func (mem Memory) result(
	items []SearchResultItem, threshold float32, limit int,
) []SearchResultItem {
	var n = limit
	if n == 0 {
		n = len(items)
	}
	results := make([]SearchResultItem, 0, n)
	sort.Slice(items, func(i, j int) bool {
		return items[i].Score > items[j].Score
	})
	for i := 0; i < len(items) &&
		items[i].Score >= threshold &&
		(limit == 0 || len(results) < limit); i++ {
		results = append(results, SearchResultItem{Index: items[i].Index, Score: items[i].Score})
	}
	return results
}

func (mem *Memory) F1(f1, f2 float32) float32 { return f1 * f2 }

func (mem *Memory) Search(
	ctx context.Context, _key Key, bs []byte,
	length uint64, threshold float32, limit int,
) ([]SearchResult, error) {

	var xl = xlog.FromContextSafe(ctx)
	_ = xl

	n := len(bs) / int(length)
	ds := make([][]SearchResultItem, n)

	_node, err := mem.cached.get(ctx, _key)
	if err != nil {
		return nil, errors.Wrapf(err, "cache get failed. %s", _key.Key())
	}

	for i := 0; i < n; i++ {
		var fs = make([]float32, _node.length/length)
		distance.DistancesCosineCgoFlat(
			bs[i*int(length):(i+1)*int(length)],
			mem.bs[_node.offset:_node.offset+_node.length],
			fs,
		)
		ds[i] = make([]SearchResultItem, _node.length/length)
		for j := 0; j < len(fs); j++ {
			ds[i][j].Index = j
			ds[i][j].Score = fs[j]
		}
	}

	_node.pending ^= PENDING_READ // TODO 并行问题

	results := make([]SearchResult, n)
	for k := 0; k < n; k++ {
		results[k].Items = mem.result(ds[k], threshold, limit)
		// xl.Infof("SEARCH: %d %d %f %d", k, len(ds[k].R), threshold, limit)
	}

	return results, nil
}
