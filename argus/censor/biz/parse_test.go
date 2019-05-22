package biz

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"qiniu.com/argus/utility/censor"
)

func TestParsePulp(t *testing.T) {
	parse := func(resp ImagePulpResp, thresholds PulpThreshold) Suggestion {
		suggestion, _ := ParsePulp(resp, thresholds)
		return suggestion
	}
	assert.Equal(t, PASS, parse(ImagePulpResp{Label: 2, Score: 1.0, Review: false}, PulpThreshold{}))
	assert.Equal(t, PASS, parse(ImagePulpResp{Label: 2, Score: 1.0, Review: true}, PulpThreshold{}))
	assert.Equal(t, PASS, parse(ImagePulpResp{Label: 2, Score: 0.7, Review: false}, PulpThreshold{Normal: newFloat32(0.6)}))
	assert.Equal(t, PASS, parse(ImagePulpResp{Label: 2, Score: 0.7, Review: true}, PulpThreshold{Normal: newFloat32(0.6)}))
	assert.Equal(t, REVIEW, parse(ImagePulpResp{Label: 2, Score: 0.5, Review: false}, PulpThreshold{Normal: newFloat32(0.6)}))
	assert.Equal(t, REVIEW, parse(ImagePulpResp{Label: 2, Score: 0.5, Review: true}, PulpThreshold{Normal: newFloat32(0.6)}))
	assert.Equal(t, REVIEW, parse(ImagePulpResp{Label: 1, Score: 1.0, Review: false}, PulpThreshold{}))
	assert.Equal(t, REVIEW, parse(ImagePulpResp{Label: 1, Score: 1.0, Review: true}, PulpThreshold{}))
	assert.Equal(t, REVIEW, parse(ImagePulpResp{Label: 1, Score: 0.7, Review: false}, PulpThreshold{Sexy: newFloat32(0.8)}))
	assert.Equal(t, REVIEW, parse(ImagePulpResp{Label: 1, Score: 0.7, Review: true}, PulpThreshold{Sexy: newFloat32(0.8)}))
	assert.Equal(t, BLOCK, parse(ImagePulpResp{Label: 1, Score: 0.9, Review: false}, PulpThreshold{Sexy: newFloat32(0.8)}))
	assert.Equal(t, BLOCK, parse(ImagePulpResp{Label: 1, Score: 0.9, Review: true}, PulpThreshold{Sexy: newFloat32(0.8)}))
	assert.Equal(t, REVIEW, parse(ImagePulpResp{Label: 0, Score: 0.0, Review: true}, PulpThreshold{}))
	assert.Equal(t, REVIEW, parse(ImagePulpResp{Label: 0, Score: 0.4, Review: true}, PulpThreshold{Pulp: newFloat32(0.5)}))
	assert.Equal(t, REVIEW, parse(ImagePulpResp{Label: 0, Score: 0.4, Review: false}, PulpThreshold{Pulp: newFloat32(0.5)}))
	assert.Equal(t, BLOCK, parse(ImagePulpResp{Label: 0, Score: 0.6, Review: true}, PulpThreshold{Pulp: newFloat32(0.5)}))
	assert.Equal(t, BLOCK, parse(ImagePulpResp{Label: 0, Score: 0.6, Review: false}, PulpThreshold{Pulp: newFloat32(0.5)}))
	assert.Equal(t, BLOCK, parse(ImagePulpResp{Label: 0, Score: 1.0, Review: false}, PulpThreshold{}))
}

func TestParseTerror(t *testing.T) {
	parse := func(resp ImageTerrorResp, thresholds TerrorThreshold) Suggestion {
		suggestion, _ := ParseTerror(resp, thresholds)
		return suggestion
	}
	assert.Equal(t, PASS, parse(ImageTerrorResp{Label: 0, Score: 1.0, Review: false}, TerrorThreshold{}))
	assert.Equal(t, PASS, parse(ImageTerrorResp{Label: 0, Score: 0.0, Review: true}, TerrorThreshold{}))
	assert.Equal(t, PASS, parse(ImageTerrorResp{Label: 0, Score: 0.6, Review: true}, TerrorThreshold{Normal: newFloat32(0.5)}))
	assert.Equal(t, PASS, parse(ImageTerrorResp{Label: 0, Score: 0.6, Review: false}, TerrorThreshold{Normal: newFloat32(0.5)}))
	assert.Equal(t, REVIEW, parse(ImageTerrorResp{Label: 0, Score: 0.4, Review: true}, TerrorThreshold{Normal: newFloat32(0.5)}))
	assert.Equal(t, REVIEW, parse(ImageTerrorResp{Label: 0, Score: 0.4, Review: false}, TerrorThreshold{Normal: newFloat32(0.5)}))
	assert.Equal(t, BLOCK, parse(ImageTerrorResp{Label: 1, Score: 1.0, Review: false}, TerrorThreshold{}))
	assert.Equal(t, REVIEW, parse(ImageTerrorResp{Label: 1, Score: 0.0, Review: true}, TerrorThreshold{}))
	assert.Equal(t, BLOCK, parse(ImageTerrorResp{Label: 1, Score: 0.6, Review: true}, TerrorThreshold{Terror: newFloat32(0.5)}))
	assert.Equal(t, BLOCK, parse(ImageTerrorResp{Label: 1, Score: 0.6, Review: false}, TerrorThreshold{Terror: newFloat32(0.5)}))
	assert.Equal(t, REVIEW, parse(ImageTerrorResp{Label: 1, Score: 0.4, Review: true}, TerrorThreshold{Terror: newFloat32(0.5)}))
	assert.Equal(t, REVIEW, parse(ImageTerrorResp{Label: 1, Score: 0.4, Review: false}, TerrorThreshold{Terror: newFloat32(0.5)}))
}

func TestParsePolitician(t *testing.T) {
	parse := func(name string, score float32, review bool, thresholds PoliticianThreshold) Suggestion {
		var resp = ImagePoliticianResp{}
		resp.Detections = append(resp.Detections, censor.FaceSearchDetail{Value: struct {
			Name   string  `json:"name,omitempty"`
			Group  string  `json:"group,omitempty"`
			Score  float32 `json:"score"`
			Review bool    `json:"review"`
		}{
			Name: name, Score: score, Review: review,
		}})
		suggestion, _ := ParsePolitician(resp, thresholds)
		return suggestion
	}
	assert.Equal(t, PASS, parse("", 1.0, false, PoliticianThreshold{}))
	assert.Equal(t, PASS, parse("", 0.0, true, PoliticianThreshold{}))
	assert.Equal(t, BLOCK, parse("X", 1.0, false, PoliticianThreshold{}))
	assert.Equal(t, REVIEW, parse("X", 0.0, true, PoliticianThreshold{}))
}
