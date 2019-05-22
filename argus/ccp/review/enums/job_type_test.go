package enums

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestJobTypeIsValid(t *testing.T) {
	assertion := assert.New(t)
	assertion.True(JobTypeStream.IsValid())
	assertion.True(JobTypeBatch.IsValid())
	assertion.False(JobType("invalid").IsValid())
}
