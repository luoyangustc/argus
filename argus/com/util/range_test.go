package util

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestParseRange(t *testing.T) {
	_, e := ParseRange("")
	assert.Error(t, e)

	_, e = ParseRange("0x-1")
	assert.Error(t, e)

	_, e = ParseRange("100")
	assert.Error(t, e)

	_, e = ParseRange("1-123456789012345678901")
	assert.Error(t, e)

	r, e := ParseRange("-")
	assert.NoError(t, e)
	assert.True(t, r.Valid())
	assert.Nil(t, r.from)
	assert.Nil(t, r.to)

	r, e = ParseRange("1-")
	assert.NoError(t, e)
	assert.True(t, r.Valid())
	assert.NotNil(t, r.from)
	assert.EqualValues(t, 1, *r.from)
	assert.Nil(t, r.to)

	r, e = ParseRange("-9")
	assert.NoError(t, e)
	assert.True(t, r.Valid())
	assert.NotNil(t, r.to)
	assert.EqualValues(t, 9, *r.to)
	assert.Nil(t, r.from)

	r, e = ParseRange("1-9")
	assert.NoError(t, e)
	assert.True(t, r.Valid())
	assert.NotNil(t, r.from)
	assert.NotNil(t, r.to)
	assert.EqualValues(t, 1, *r.from)
	assert.EqualValues(t, 9, *r.to)

	r, e = ParseRange("-9--1")
	assert.NoError(t, e)
	assert.True(t, r.Valid())
	assert.NotNil(t, r.from)
	assert.NotNil(t, r.to)
	assert.EqualValues(t, -9, *r.from)
	assert.EqualValues(t, -1, *r.to)

	r, e = ParseRange("9-1")
	assert.NoError(t, e)
	assert.False(t, r.Valid())
	r, e = ParseRange("-1--9")
	assert.NoError(t, e)
	assert.False(t, r.Valid())
}

func TestRangeLimit(t *testing.T) {
	r := NewRange(Int64Ptr(1), Int64Ptr(9))
	assert.True(t, r.Valid())
	assert.Equal(t, "1-9", r.String())
	assert.EqualValues(t, 9, r.Count())

	r.SetLimit(Int64Ptr(2), Int64Ptr(8))
	assert.True(t, r.Valid())
	assert.Equal(t, "2-8", r.String())
	assert.EqualValues(t, 7, r.Count())

	r.SetLimit(Int64Ptr(0), Int64Ptr(10))
	assert.True(t, r.Valid())
	assert.Equal(t, "1-9", r.String())
	assert.EqualValues(t, 9, r.Count())

	r.SetLimit(Int64Ptr(-9), Int64Ptr(-1))
	assert.True(t, r.Valid())
	assert.Empty(t, r.String())
	assert.EqualValues(t, 0, r.Count())

	r.SetLimit(Int64Ptr(-5), Int64Ptr(5))
	assert.True(t, r.Valid())
	assert.Equal(t, "1-5", r.String())
	assert.EqualValues(t, 5, r.Count())

	r.SetLimit(Int64Ptr(5), Int64Ptr(15))
	assert.True(t, r.Valid())
	assert.Equal(t, "5-9", r.String())
	assert.EqualValues(t, 5, r.Count())

	r.SetLimit(Int64Ptr(10), Int64Ptr(15))
	assert.True(t, r.Valid())
	assert.Empty(t, r.String())
	assert.EqualValues(t, 0, r.Count())

	r = NewRange(nil, nil)
	assert.True(t, r.Valid())
	assert.Equal(t, "-", r.String())
	r.SetLimit(Int64Ptr(1), nil)
	assert.True(t, r.Valid())
	assert.Equal(t, "1-", r.String())
	r.SetLimit(nil, Int64Ptr(9))
	assert.True(t, r.Valid())
	assert.Equal(t, "1-9", r.String())
}
