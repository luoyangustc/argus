package enums

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestSuggestionIsValid(t *testing.T) {
	assertion := assert.New(t)

	assertion.True(SuggestionPass.IsValid())
	assertion.True(SuggestionReview.IsValid())
	assertion.True(SuggestionBlock.IsValid())
	assertion.True(SuggestionDisabled.IsValid())
	assertion.False(Suggestion("invalid").IsValid())
}

func TestSuggestionIsAttention(t *testing.T) {
	assertion := assert.New(t)

	assertion.False(SuggestionPass.IsAttention())
	assertion.False(SuggestionReview.IsAttention())
	assertion.True(SuggestionBlock.IsAttention())
	assertion.True(SuggestionDisabled.IsAttention())
	assertion.False(Suggestion("invalid").IsAttention())
}
