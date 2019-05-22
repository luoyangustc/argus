package review

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestNewReviewService(t *testing.T) {
	assertion := assert.New(t)
	assertion.NotNil(NewService())
}
