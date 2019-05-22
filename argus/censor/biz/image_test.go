package biz

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"qiniu.com/argus/utility/censor"
)

func TestParseImagePoliticianResp(t *testing.T) {

	t.Run("", func(t *testing.T) {
		resp := ParseImagePoliticianResp(
			ImagePoliticianResp{Detections: []censor.FaceSearchDetail{
				func() (ret censor.FaceSearchDetail) {
					ret.Value.Name = ""
					return
				}(),
			}},
			PoliticianThreshold{},
		)
		assert.Equal(t, PASS, resp.Suggestion)
	})

	t.Run("", func(t *testing.T) {
		resp := ParseImagePoliticianResp(
			ImagePoliticianResp{Detections: []censor.FaceSearchDetail{
				func() (ret censor.FaceSearchDetail) {
					ret.Value.Name = ""
					return
				}(),
				func() (ret censor.FaceSearchDetail) {
					ret.Value.Name = "X"
					return
				}(),
			}},
			PoliticianThreshold{},
		)
		assert.Equal(t, BLOCK, resp.Suggestion)
	})
}
