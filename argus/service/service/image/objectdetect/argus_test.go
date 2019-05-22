package objectdetect

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	pimage "qiniu.com/argus/service/service/image"
)

var eods = EvalObjectDetectEndpoints{EvalObjectDetectEP: func(ctx context.Context, request interface{}) (interface{}, error) {
	return EvalObjectDetectResp{
		Code:    0,
		Message: "",
		Result: struct {
			Detections []EvalObjectDetect `json:"detections"`
		}{
			Detections: []EvalObjectDetect{
				EvalObjectDetect{
					Index:   1,
					Class:   "glass",
					Score:   0.99,
					Pts:     [][2]int{{1, 2}, {3, 4}},
					LabelCN: "眼镜",
				},
			},
		},
	}, nil
},
}

func TestObjectDetect(t *testing.T) {
	s, _ := NewObjectDetectService(eods)
	resp, err := s.ObjectDetect(context.Background(), DetectionReq{
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

	pts := [][2]int{{1, 2}, {3, 4}}
	assert.Nil(t, err)
	assert.Equal(t, 0, resp.Code)
	assert.Equal(t, 1, resp.Result.Detections[0].Index)
	assert.Equal(t, float32(0.99), resp.Result.Detections[0].Score)
	assert.Equal(t, "glass", resp.Result.Detections[0].Class)
	assert.Equal(t, pts, resp.Result.Detections[0].Pts)
	assert.Equal(t, "眼镜", resp.Result.Detections[0].LabelCN)
}
