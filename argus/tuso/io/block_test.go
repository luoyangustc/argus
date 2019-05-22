package io

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestBlocks(t *testing.T) {
	var (
		ctx    = context.Background()
		blocks = NewMockBlocks(4, 4)
	)

	{
		s := blocks.NewScanner(ctx, 0, 0)
		assert.False(t, s.Scan(ctx))
	}
	{
		w := blocks.NewWriter(ctx, 0)
		w.Write(ctx, []byte{1, 0, 0, 0})
		w.Close(ctx)
	}
	{
		s := blocks.NewScanner(ctx, 0, 0)
		assert.True(t, s.Scan(ctx))
		bs := s.Bytes(ctx)
		assert.Equal(t, byte(1), bs[0])
	}
	{
		w := blocks.NewWriter(ctx, 1)
		w.Write(ctx, []byte{2, 0, 0, 0})
		w.Write(ctx, []byte{3, 0, 0, 0})
		w.Write(ctx, []byte{4, 0, 0, 0})
		w.Write(ctx, []byte{5, 0, 0, 0})
		w.Close(ctx)
	}
	t.Logf("%v", blocks.(*MockBlocks).bytes)
	{
		s := blocks.NewScanner(ctx, 0, 4)
		var bs []byte
		assert.True(t, s.Scan(ctx))
		bs = s.Bytes(ctx)
		assert.Equal(t, byte(1), bs[0])
		assert.True(t, s.Scan(ctx))
		bs = s.Bytes(ctx)
		assert.Equal(t, byte(2), bs[0])
		assert.True(t, s.Scan(ctx))
		bs = s.Bytes(ctx)
		assert.Equal(t, byte(3), bs[0])
		assert.True(t, s.Scan(ctx))
		bs = s.Bytes(ctx)
		assert.Equal(t, byte(4), bs[0])
		assert.False(t, s.Scan(ctx))
	}
	{
		s := blocks.NewScanner(ctx, 4, 0)
		assert.True(t, s.Scan(ctx))
		bs := s.Bytes(ctx)
		assert.Equal(t, byte(5), bs[0])
		assert.False(t, s.Scan(ctx))
	}
}
