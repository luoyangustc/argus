package records

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestMock(t *testing.T) {

	ctx := context.Background()
	rs := NewMock()

	{
		count, _ := rs.Count(ctx)
		assert.Equal(t, count, 0)
		ok, _ := rs.HasKey(ctx, "A")
		assert.False(t, ok)
		ok, _ = rs.HasKey(ctx, "B")
		assert.False(t, ok)
	}
	{
		_ = rs.Append(ctx, "A", []byte{1})
		count, _ := rs.Count(ctx)
		assert.Equal(t, count, 1)
		ok, _ := rs.HasKey(ctx, "A")
		assert.True(t, ok)
		ok, _ = rs.HasKey(ctx, "B")
		assert.False(t, ok)
	}
	{
		_ = rs.Append(ctx, "B", []byte{1})
		count, _ := rs.Count(ctx)
		assert.Equal(t, count, 2)
		ok, _ := rs.HasKey(ctx, "A")
		assert.True(t, ok)
		ok, _ = rs.HasKey(ctx, "B")
		assert.True(t, ok)
	}
}
