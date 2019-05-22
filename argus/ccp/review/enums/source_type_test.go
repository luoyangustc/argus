package enums

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestSourceTypeIsValid(t *testing.T) {
	assertion := assert.New(t)

	assertion.True(SourceTypeKodo.IsValid())
	assertion.True(SourceTypeApi.IsValid())
	assertion.False(SourceType("invalid").IsValid())
}
