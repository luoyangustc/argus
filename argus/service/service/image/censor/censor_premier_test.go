package censor

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	pimage "qiniu.com/argus/service/service/image"
)

func TestPremierCensor(t *testing.T) {
	s, _ := NewCensorService(DEFAULT, epps, epfs, efds, effs, eps, etms, etds, eaqs, eads, ears, eacs)

	resp, err := s.PremierCensor(context.Background(), IPremierCensorRequest{
		Datas: []struct {
			DataID string       `json:"data_id,omitempty"`
			IMG    pimage.Image `json:"-"`
			URI    string       `json:"uri"`
		}{
			struct {
				DataID string       `json:"data_id,omitempty"`
				IMG    pimage.Image `json:"-"`
				URI    string       `json:"uri"`
			}{
				URI: "data:application/octet-stream;base64,/9j/4AAQ",
			},
		},
		Params: struct {
			Scenes       []string                   `json:"scenes,omitempty"`
			ScenesParams map[string]json.RawMessage `json:"scenes_params"`
		}{
			Scenes: []string{"pulp", "terror", "politician", "ads"},
		},
	})

	assert.Nil(t, err)
	assert.Equal(t, pimage.BLOCK, resp.Result.Suggestion)

	repPulp := resp.Result.Scenes[PULP]
	repTerror := resp.Result.Scenes[TERROR]
	repPolitician := resp.Result.Scenes[POLITICIAN]
	repTerrorResult := repTerror.Details[0]
	repAds := resp.Result.Scenes[ADS]

	assert.Equal(t, pimage.PASS, repPulp.Suggestion)
	assert.Equal(t, "normal", repPulp.Details[0].Label)
	assert.Equal(t, string(pimage.REVIEW), string(repTerror.Suggestion))
	assert.Equal(t, float32(0.99), repTerrorResult.Score)
	assert.Equal(t, pimage.BLOCK, repPolitician.Suggestion)
	assert.Equal(t, float32(0.998), repPolitician.Details[0].Detections[0].Score)
	assert.Equal(t, "XXX", repPolitician.Details[0].Label)
	assert.Equal(t, "affairs_official_gov", repPolitician.Details[0].Group)

	assert.Equal(t, string(pimage.REVIEW), string(repAds.Suggestion))
	assert.Equal(t, "ads", repAds.Details[0].Label)
	assert.Equal(t, pimage.REVIEW, repAds.Details[0].Suggestion)
	assert.Equal(t, float32(0.5), repAds.Details[0].Score)
}
