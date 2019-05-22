package biz

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestSuggestion(t *testing.T) {
	assert.Equal(t, BLOCK, Suggestion("").Update(BLOCK))
	assert.Equal(t, REVIEW, Suggestion("").Update(REVIEW))
	assert.Equal(t, PASS, Suggestion("").Update(PASS))
	assert.Equal(t, BLOCK, BLOCK.Update(BLOCK))
	assert.Equal(t, BLOCK, BLOCK.Update(REVIEW))
	assert.Equal(t, BLOCK, BLOCK.Update(PASS))
	assert.Equal(t, BLOCK, REVIEW.Update(BLOCK))
	assert.Equal(t, REVIEW, REVIEW.Update(REVIEW))
	assert.Equal(t, REVIEW, REVIEW.Update(PASS))
	assert.Equal(t, BLOCK, PASS.Update(BLOCK))
	assert.Equal(t, REVIEW, PASS.Update(REVIEW))
	assert.Equal(t, PASS, PASS.Update(PASS))
}
