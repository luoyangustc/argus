package feature_group

import (
	"container/list"
	"context"
	"math/rand"
	"sync"

	"github.com/pkg/errors"
)

type Key interface {
	Key() string
}

const (
	PENDING_READ  byte = 0x01
	PENDING_WRITE byte = 0x02
)

type node struct {
	offset, size uint64
	chunk        *list.Element

	pending byte
	cleanup func()

	length uint64
}

func (n *node) Cleanup() {
	if n.cleanup != nil {
		n.cleanup()
	}
}

type cached struct {
	mutex   *sync.RWMutex
	m       map[string]*nodeElement
	storage *storage
	fetch   func(context.Context, Key) ([]byte, error)
	set     func(context.Context, uint64, []byte) error
}

func newCached(config StorageConfig,
	fetch func(context.Context, Key) ([]byte, error),
	set func(context.Context, uint64, []byte) error,
) *cached {
	return &cached{
		mutex:   new(sync.RWMutex),
		m:       make(map[string]*nodeElement),
		storage: newStorage(config),
		fetch:   fetch,
		set:     set,
	}
}

func (c *cached) get(ctx context.Context, _key Key) (*node, error) {
	c.mutex.Lock()
	_node, ok := c.m[_key.Key()]
	if ok {
		_CacheHitGauge(_node.Node().size).Inc()
		_node.Node().pending |= PENDING_READ
		c.mutex.Unlock()
		return _node.Node(), nil
	}
	_CacheMissGauge(0).Inc()
	c.mutex.Unlock()

	bs, err := c.fetch(ctx, _key)
	if err != nil {
		return nil, err
	}
	c.mutex.Lock()
	_node, ok = c.m[_key.Key()]
	if ok {
		_node.Node().pending |= PENDING_READ
		c.mutex.Unlock()
		return _node.Node(), nil
	}
	_node = c.storage.set(uint64(len(bs)), func() { delete(c.m, _key.Key()) })
	if _node == nil {
		c.mutex.Unlock()
		return nil, nil
	}
	_node.Node().pending |= PENDING_READ
	c.m[_key.Key()] = _node
	c.mutex.Unlock()
	err = c.set(ctx, _node.Node().offset, bs)
	if err != nil {
		return nil, errors.Wrap(err, "cached.set")
	}
	_node.Node().pending ^= PENDING_WRITE
	return _node.Node(), nil
}

////////////////////////////////////////////////////////////////////////////////

type chunk struct {
	offset, size uint64
}

type StorageConfig struct {
	Size      uint64 `json:"size"`
	ChunkSize uint64 `json:"chunk_size"`
	Slabs     []struct {
		Size  uint64 `json:"size"`
		Count int    `json:"count"`
	} `json:"slabs"`
}

type storage struct {
	StorageConfig
	slabs  []*slabs
	chunks *list.List
}

func newStorage(config StorageConfig) *storage {

	chunks := list.New()
	var offset uint64
	for ; offset+config.ChunkSize <= config.Size; offset += config.ChunkSize {
		chunks.PushBack(chunk{offset: offset, size: config.ChunkSize})
	}
	slabs := make([]*slabs, 0, len(config.Slabs))
	{
		e := chunks.Front()
		for _, slabConfig := range config.Slabs {
			_slabs := newSlabs(config.Size, slabConfig.Size, nil)
			for ; e != nil && _slabs.count() < slabConfig.Count; e = e.Next() {
				_slabs.add(e.Value.(chunk).offset, e.Value.(chunk).size, e)
			}
			slabs = append(slabs, _slabs)
		}
	}

	return &storage{StorageConfig: config, slabs: slabs, chunks: chunks}
}

func (s *storage) set(length uint64, clean func()) *nodeElement {
	i, slabs := 0, s.slabs[0]
	for ; i < len(s.slabs) && length > slabs.size; i++ {
		if i+1 < len(s.slabs) {
			slabs = s.slabs[i+1]
		}
	}
	if i >= len(s.slabs) {
		return nil // TODO TooBig
	}
	// TODO take other chunks
	return slabs.set(length, clean)
}

////////////////////////////////////////////////////////////////////////////////

type nodeElement list.Element

func (e *nodeElement) Node() *node {
	if e.Value == nil {
		return nil
	}
	return e.Value.(*node)
}

type slabs struct {
	mutex *sync.RWMutex
	size  uint64
	list  *list.List
	heap  *slabList
}

func newSlabs(size, slabSize uint64, mutex *sync.RWMutex) *slabs {
	heap := newSlabList(size, mutex)
	return &slabs{
		mutex: mutex,
		size:  slabSize,
		heap:  heap,
		list:  list.New(),
	}
}

func (ss *slabs) count() int { return ss.list.Len() }

func (ss *slabs) add(offset, size uint64, chunk *list.Element) {
	var initOffset = offset
	for ; offset+ss.size <= initOffset+size; offset += ss.size {
		node := &node{offset: offset, size: ss.size, chunk: chunk}
		ss.heap.dirtyAdd(node)
		ss.list.PushFront(node)
	}
}

func (ss *slabs) remove() *list.Element {
	// TODO
	// for offset := from; offset+ss.size < to; offset += ss.size {
	// 	ss.list.dirtyRemove(&slabListItem{node: &node{offset: offset, size: ss.size}})
	// }
	return nil
}

func (ss *slabs) set(length uint64, cleanup func()) *nodeElement {
	if ss.list.Len() == 0 {
		return nil
	}
	for e := ss.list.Front(); e != nil; e = e.Next() {
		node := (*nodeElement)(e)
		if node.Node().pending == 0 {
			ss.list.MoveToBack(e)
			node.Node().Cleanup()
			node.Node().pending |= PENDING_WRITE
			node.Node().length = length
			node.Node().cleanup = cleanup
			return node
		}
	}
	return nil // OutOfSpace
}

func (ss *slabs) update(node *nodeElement) {
	ss.list.MoveToBack((*list.Element)(node))
}

////////////////////////////////////////////////////////////////////////////////

const (
	skiplistHeight = 5
)

type slabList struct {
	mutex *sync.RWMutex
	head  *slabListItem
}

type slabListItem struct {
	node *node

	nexts [skiplistHeight]*slabListItem
}

func (item *slabListItem) next() *slabListItem { return item.nexts[0] }

func newSlabList(size uint64, mutex *sync.RWMutex) *slabList {
	head := &slabListItem{node: &node{offset: 0, length: 0}}
	tail := &slabListItem{node: &node{offset: size, length: 0}}
	for i := 0; i < skiplistHeight; i++ {
		head.nexts[i] = tail
	}
	return &slabList{
		mutex: mutex,
		head:  head,
	}
}

func (l *slabList) dirtyFindPrevs(offset uint64) (prevs [skiplistHeight]*slabListItem) {
	for prev, height := l.head, skiplistHeight-1; height >= 0; height-- {
		var next *slabListItem
		for {
			next = prev.nexts[height]
			if offset <= next.node.offset {
				break
			}
			prev = next
		}
		prevs[height] = prev
	}
	return
}

func (l *slabList) findPrevs(offset uint64) (prevs [skiplistHeight]*slabListItem) {
	l.mutex.Lock()
	defer l.mutex.Unlock()
	return l.dirtyFindPrevs(offset)
}

func randHeight() int {
	h := 0
	for h < skiplistHeight-1 && rand.Intn(2) == 0 {
		h++
	}
	return h
}

func (l *slabList) dirtyAdd(node *node) {
	item := &slabListItem{node: node}
	prevs := l.dirtyFindPrevs(item.node.offset)
	h := randHeight()
	for h >= 0 {
		prev := prevs[h]
		item.nexts[h] = prev.nexts[h]
		prev.nexts[h] = item
		h--
	}
}

func (l *slabList) add(node *node) {
	l.mutex.Lock()
	defer l.mutex.Unlock()
	l.dirtyAdd(node)
}

func (l *slabList) dirtyRemove(offset uint64) {
	prevs := l.dirtyFindPrevs(offset)
	item := prevs[0].next()
	for h := 0; h <= skiplistHeight-1; h++ {
		if item.nexts[h] != nil {
			prevs[h].nexts[h] = item.nexts[h]
		}
	}
}

func (l *slabList) remove(offset uint64) {
	l.mutex.Lock()
	defer l.mutex.Unlock()
	l.dirtyRemove(offset)
}
