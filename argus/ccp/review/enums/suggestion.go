package enums

// Suggestion define
type Suggestion string

const (
	SuggestionPass     Suggestion = "PASS"
	SuggestionReview   Suggestion = "REVIEW"
	SuggestionBlock    Suggestion = "BLOCK"
	SuggestionDisabled Suggestion = "DISABLED"
)

func (this Suggestion) IsValid() bool {
	switch this {
	case SuggestionPass, SuggestionReview, SuggestionBlock, SuggestionDisabled:
		return true
	default:
		return false
	}
}

func (this Suggestion) IsAttention() bool {
	switch this {
	case SuggestionBlock, SuggestionDisabled:
		return true
	default:
		return false
	}
}
