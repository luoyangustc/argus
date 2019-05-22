package pulp

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	pimage "qiniu.com/argus/service/service/image"
)

func TestPulpCensor(t *testing.T) {
	config := Config{
		PulpReviewThreshold: 0.6,
		SugConfig: pimage.SugConfig{
			Rules: map[string]pimage.RuleConfig{
				"sexy": pimage.RuleConfig{
					SureSuggestion:   pimage.REVIEW,
					UnsureSuggestion: pimage.REVIEW,
				},
			},
		},
	}
	s1, _ := NewPulpService(config, epps, epss1)
	resp1, err := s1.PulpCensor(context.Background(), pimage.ImageCensorReq{})

	assert.NoError(t, err)
	assert.Equal(t, pimage.PASS, resp1.Suggestion)
	assert.Equal(t, "normal", resp1.Details[0].Label)
	assert.Equal(t, pimage.PASS, resp1.Details[0].Suggestion)
	assert.Equal(t, float32(0.59), resp1.Details[0].Score)

	s2, _ := NewPulpService(config, epps, epss2)
	resp2, err := s2.PulpCensor(context.Background(), pimage.ImageCensorReq{})

	assert.NoError(t, err)
	assert.Equal(t, "review", string(resp2.Suggestion))
	assert.Equal(t, "sexy", resp2.Details[0].Label)

	config = Config{
		PulpReviewThreshold: 0.6,
		SugConfig: pimage.SugConfig{
			Rules: map[string]pimage.RuleConfig{
				"pulp": pimage.RuleConfig{
					SureSuggestion:   pimage.BLOCK,
					UnsureSuggestion: pimage.REVIEW,
				},
			},
		},
	}
	s3, _ := NewPulpService(config, epps2, epss2)
	resp3, err := s3.PulpCensor(context.Background(), pimage.ImageCensorReq{})
	assert.NoError(t, err)
	assert.Equal(t, "block", string(resp3.Suggestion))
	assert.Equal(t, "pulp", resp3.Details[0].Label)

}
