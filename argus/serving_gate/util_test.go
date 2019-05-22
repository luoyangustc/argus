package gate

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestP2S(t *testing.T) {
	assert.Equal(t, "nil", p2s(nil))
	var a = "A"
	assert.Equal(t, "A", p2s(&a))
}

func TestParseOP(t *testing.T) {
	{
		_, _, err := parseOP("/eval/xxx")
		assert.Error(t, ErrBadOP, err)
	}
	{
		cmd, version, err := parseOP("/v1/eval/foo")
		assert.NoError(t, err)
		assert.Equal(t, "foo", cmd)
		assert.Nil(t, version)
	}
	{
		cmd, version, err := parseOP("/v1/eval/foo/11")
		assert.NoError(t, err)
		assert.Equal(t, "foo", cmd)
		assert.Equal(t, "11", *version)
	}
	{
		cmd, _, err := parseOP("/v1/eval/ava-classify")
		assert.NoError(t, err)
		assert.Equal(t, "ava-classify", cmd)
	}
}
