package jupyter

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestSecret(t *testing.T) {
	s := Secret{
		Algorithm: secretAlgorithm,
		Salt:      "01234567",
		phrase:    "text",
	}
	assert.Equal(t, "sha1:01234567:3da60db65aa3df4b26066060156fab76526d3bd7", s.String())
}
