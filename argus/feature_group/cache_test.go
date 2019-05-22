package feature_group

import (
	"math/rand"
	"testing"

	"github.com/stretchr/testify/assert"
)

//----------------------------------------------------------------------------//

func TestStorage(t *testing.T) {
	config := StorageConfig{Size: 16, ChunkSize: 4}
	config.Slabs = append(config.Slabs,
		struct {
			Size  uint64 `json:"size"`
			Count int    `json:"count"`
		}{Size: 2, Count: 4},
		struct {
			Size  uint64 `json:"size"`
			Count int    `json:"count"`
		}{Size: 4, Count: 2})
	s := newStorage(config)

	assert.Nil(t, s.set(6, nil))

	assert.Equal(t, uint64(4), s.set(3, nil).Node().size)
	assert.Equal(t, uint64(2), s.set(1, nil).Node().size)
	assert.Equal(t, uint64(4), s.set(4, nil).Node().size)
	assert.Equal(t, uint64(2), s.set(2, nil).Node().size)
	assert.Nil(t, s.set(3, nil))
	assert.Equal(t, uint64(2), s.set(1, nil).Node().size)
	assert.Equal(t, uint64(2), s.set(1, nil).Node().size)
	assert.Nil(t, s.set(1, nil))

}

//----------------------------------------------------------------------------//

func TestSlabs(t *testing.T) {
	slabs := newSlabs(16, 2, nil)

	assert.Nil(t, slabs.set(1, nil))

	slabs.add(4, 4, nil)
	node1 := slabs.set(1, nil)
	node2 := slabs.set(1, nil)
	assert.Equal(t, uint64(2), node1.Node().size)
	assert.Equal(t, PENDING_WRITE, node1.Node().pending&PENDING_WRITE)
	assert.Equal(t, uint64(2), node2.Node().size)
	assert.Equal(t, PENDING_WRITE, node2.Node().pending&PENDING_WRITE)

	assert.Nil(t, slabs.set(1, nil))
	node1.Node().pending = 0x0
	node1 = slabs.set(1, nil)
	assert.Equal(t, uint64(2), node1.Node().size)
	assert.Equal(t, PENDING_WRITE, node1.Node().pending&PENDING_WRITE)

	assert.Nil(t, slabs.set(1, nil))
	slabs.add(8, 4, nil)
	node3 := slabs.set(1, nil)
	node4 := slabs.set(1, nil)
	assert.Equal(t, uint64(2), node3.Node().size)
	assert.Equal(t, PENDING_WRITE, node3.Node().pending&PENDING_WRITE)
	assert.Equal(t, uint64(2), node4.Node().size)
	assert.Equal(t, PENDING_WRITE, node4.Node().pending&PENDING_WRITE)

}

func BenchmarkSlabs(b *testing.B) {
	slabs := newSlabs(1024, 2, nil)
	slabs.add(0, 1024, nil)
	for i := 0; i < b.N; i++ {
		node1 := slabs.set(1, nil)
		node2 := slabs.set(1, nil)
		node1.Node().pending = 0x0
		node2.Node().pending = 0x0
	}
}

// ---------------------------------------------------------------------------//

func TestSlabList(t *testing.T) {

	list := newSlabList(16, nil)

	assert.Equal(t, uint64(0), list.dirtyFindPrevs(4)[0].node.offset)
	assert.Equal(t, uint64(0), list.dirtyFindPrevs(4)[0].node.size)

	list.dirtyAdd(&node{offset: 2, size: 2})
	assert.Equal(t, uint64(0), list.dirtyFindPrevs(2)[0].node.offset)
	assert.Equal(t, uint64(0), list.dirtyFindPrevs(2)[0].node.size)
	assert.Equal(t, uint64(2), list.dirtyFindPrevs(4)[0].node.offset)
	assert.Equal(t, uint64(2), list.dirtyFindPrevs(4)[0].node.size)

	list.dirtyAdd(&node{offset: 6, size: 2})
	assert.Equal(t, uint64(2), list.dirtyFindPrevs(6)[0].node.offset)
	assert.Equal(t, uint64(2), list.dirtyFindPrevs(6)[0].node.size)
	assert.Equal(t, uint64(6), list.dirtyFindPrevs(8)[0].node.offset)
	assert.Equal(t, uint64(2), list.dirtyFindPrevs(8)[0].node.size)

	list.dirtyAdd(&node{offset: 10, size: 2})
	list.dirtyRemove(6)
	assert.Equal(t, uint64(2), list.dirtyFindPrevs(6)[0].node.offset)
	assert.Equal(t, uint64(2), list.dirtyFindPrevs(6)[0].node.size)
	assert.Equal(t, uint64(2), list.dirtyFindPrevs(8)[0].node.offset)
	assert.Equal(t, uint64(2), list.dirtyFindPrevs(8)[0].node.size)
	assert.Equal(t, uint64(10), list.dirtyFindPrevs(12)[0].node.offset)
	assert.Equal(t, uint64(2), list.dirtyFindPrevs(12)[0].node.size)

}

func BenchmarkSlabList(b *testing.B) {
	list := newSlabList(1024, nil)
	for i := 0; i < 1024; i++ {
		list.dirtyAdd(&node{offset: uint64(i), size: 1})
	}
	for i := 0; i < b.N; i++ {
		list.dirtyFindPrevs(uint64(rand.Int31n(1024)))
	}
}
