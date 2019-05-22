package terror

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	pimage "qiniu.com/argus/service/service/image"
)

func TestTerrorCensor(t *testing.T) {

	config := Config{
		TerrorThreshold: 0.85,
		SugConfig: pimage.SugConfig{
			CensorBy: "label",
			Rules: map[string]pimage.RuleConfig{
				"normal":             RuleAlwaysPass,
				"guns":               RuleWithBlock,
				"self_burning":       RuleWithBlock,
				"beheaded":           RuleWithBlock,
				"illegal_flag":       RuleWithBlock,
				"fight_person":       RuleAlwaysReview,
				"fight_police":       RuleAlwaysReview,
				"anime_knives":       RuleAlwaysReview,
				"knives":             RuleAlwaysReview,
				"anime_guns":         RuleAlwaysReview,
				"bloodiness":         RuleAlwaysReview,
				"anime_bloodiness":   RuleAlwaysReview,
				"special_clothing":   RuleAlwaysReview,
				"march_crowed":       RuleAlwaysReview,
				"special_characters": RuleAlwaysReview,
			},
		},
	}

	s, _ := NewTerrorService(config, etpds, etds)
	resp, err := s.TerrorCensor(context.Background(), pimage.ImageCensorReq{})

	assert.NoError(t, err)
	assert.Equal(t, pimage.REVIEW, resp.Suggestion)
	assert.Equal(t, 3, len(resp.Details))

	assert.Equal(t, pimage.REVIEW, resp.Details[0].Suggestion)
	assert.Equal(t, "beheaded", resp.Details[0].Label)
	assert.Equal(t, float32(0.69), resp.Details[0].Score)

	assert.Equal(t, pimage.REVIEW, resp.Details[1].Suggestion)
	assert.Equal(t, "knives", resp.Details[1].Label)
	assert.Equal(t, float32(0.79), resp.Details[1].Score)
	assert.Equal(t, 2, len(resp.Details[1].Detections))
	assert.Equal(t, float32(0.79), resp.Details[1].Detections[0].Score)
	assert.Equal(t, float32(0.69), resp.Details[1].Detections[1].Score)

	assert.Equal(t, pimage.REVIEW, resp.Details[2].Suggestion)
	assert.Equal(t, "guns", resp.Details[2].Label)
	assert.Equal(t, float32(0.5), resp.Details[2].Score)
	assert.Equal(t, 1, len(resp.Details[2].Detections))
	assert.Equal(t, float32(0.5), resp.Details[2].Detections[0].Score)
}
