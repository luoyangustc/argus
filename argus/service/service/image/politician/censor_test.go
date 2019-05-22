package politician

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	pimage "qiniu.com/argus/service/service/image"
)

func TestPoliticianCensor(t *testing.T) {

	config := Config{
		SugConfig: pimage.SugConfig{
			CensorBy: "group",
			Rules: map[string]pimage.RuleConfig{
				"affairs_official_gov": pimage.RuleConfig{
					SureThreshold:    0.45,
					AbandonThreshold: 0.35,
					SureSuggestion:   pimage.BLOCK,
					UnsureSuggestion: pimage.REVIEW,
				},
			},
		},
	}
	s, _ := NewFaceSearchService(config, efds, effs, eps)
	resp, err := s.PoliticianCensor(context.Background(), pimage.ImageCensorReq{})
	assert.NoError(t, err)
	assert.Equal(t, pimage.BLOCK, resp.Suggestion)
	assert.Equal(t, float32(0.998), resp.Details[0].Score)
	assert.Equal(t, float32(0.998), resp.Details[0].Detections[0].Score)
	assert.Equal(t, "XXX", resp.Details[0].Label)
	assert.Equal(t, "affairs_official_gov", resp.Details[0].Group)
}
