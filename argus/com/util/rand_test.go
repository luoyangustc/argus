package util

import (
	"math/rand"
	"testing"

	"github.com/stretchr/testify/assert"
)

// this test case will be very useful when you need some randomness, a complex password for example.
func TestRandomString(t *testing.T) {
	s, e := RandomString(32, 62)
	assert.Nil(t, e)
	t.Log(s)
	s, e = RandomString(32, RandomAlphabetLength)
	assert.Nil(t, e)
	t.Log(s)

}

func TestRandomStringFixedSeed(t *testing.T) {
	type testT struct {
		length int
		base   int
		nuance string
	}
	tests := []testT{
		{
			length: 30,
			base:   8,
			nuance: "104227014321405137645635265246",
		},
		{
			length: 30,
			base:   10,
			nuance: "104922870143214051378649563526",
		},
		{
			length: 30,
			base:   16,
			nuance: "105f1bf10e6b2437572b9eb55daabd",
		},
		{
			length: 30,
			base:   36,
			nuance: "1gp2bz07b11rxrfnfylira2f6dyu2u",
		},
		{
			length: 30,
			base:   62,
			nuance: "1gp2HbYz07QQb11rZPxIrfBnSfylGX",
		},
		{
			length: 30,
			base:   64,
			nuance: "1EnIb3L2JvgXknWKBL6IETv6LLht9Y",
		},
	}

	for _, test := range tests {
		rand.Seed(0)
		s, e := RandomString(test.length, test.base)
		assert.Nil(t, e)
		assert.Equal(t, test.nuance, s)
	}
}

func TestBits(t *testing.T) {
	tests := map[uint64]uint64{
		0:         0,
		1:         1,
		2:         2,
		3:         2,
		4:         3,
		7:         3,
		8:         4,
		15:        4,
		16:        5,
		1<<63 - 1: 63,
		1 << 63:   64,
		1<<64 - 1: 64,
	}

	for i, b := range tests {
		assert.Equal(t, b, bits(i))
	}
}
