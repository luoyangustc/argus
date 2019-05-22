package scene

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	pimage "qiniu.com/argus/service/service/image"
)

var eods = EvalSceneEndpoints{EvalSceneEP: func(ctx context.Context, request interface{}) (interface{}, error) {
	return EvalSceneResp{
		Code:    0,
		Message: "",
		Result: struct {
			Confidences []struct {
				Class   string   `json:"class"`
				Index   int      `json:"index"`
				Label   []string `json:"label"`
				Score   float32  `json:"score"`
				LabelCN string   `json:"label_cn,omitempty"`
			} `json:"confidences"`
		}{
			Confidences: []struct {
				Class   string   `json:"class"`
				Index   int      `json:"index"`
				Label   []string `json:"label"`
				Score   float32  `json:"score"`
				LabelCN string   `json:"label_cn,omitempty"`
			}{
				{
					Index:   1,
					Class:   "glass",
					Score:   0.99,
					Label:   []string{"glass"},
					LabelCN: "眼镜",
				},
			},
		},
	}, nil
},
}

func TestScene(t *testing.T) {
	s, _ := NewSceneService(eods)
	resp, err := s.Scene(context.Background(), SceneReq{
		Data: struct {
			IMG pimage.Image
		}{
			IMG: struct {
				Format string        `json:"format"`
				Width  int           `json:"width"`
				Height int           `json:"height"`
				URI    pimage.STRING `json:"uri"`
			}{
				URI: pimage.STRING("http://test.image.jpg"),
			},
		},
	})

	assert.Nil(t, err)
	assert.Equal(t, 0, resp.Code)
	assert.Equal(t, 1, resp.Result.Confidences[0].Index)
	assert.Equal(t, float32(0.99), resp.Result.Confidences[0].Score)
	assert.Equal(t, "glass", resp.Result.Confidences[0].Class)
	assert.Equal(t, []string{"glass"}, resp.Result.Confidences[0].Label)
	assert.Equal(t, "眼镜", resp.Result.Confidences[0].LabelCN)
}
