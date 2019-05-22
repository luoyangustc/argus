package ads

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	pimage "qiniu.com/argus/service/service/image"
)

var eaqs = EvalAdsQrcodeEndpoints{EvalAdsQrcodeEP: func(ctx context.Context, request interface{}) (interface{}, error) {
	return AdsQrcodeResp{
		Code:    0,
		Message: "",
		Result: struct {
			Detections []QrcodeDetection `json:"detections"`
		}{
			Detections: []QrcodeDetection{
				{
					Class: "qr_code",
					Index: 2,
					Pts: [][2]int{
						{176, 529}, {891, 529}, {891, 1269}, {176, 1269},
					},
					Score: 0.1999904632568359,
				},
			},
		},
	}, nil
},
}

var eads = EvalAdsDetectEndpoints{EvalAdsDetectEP: func(ctx context.Context, request interface{}) (interface{}, error) {
	return AdsDetectionResp{
		Code:    0,
		Message: "",
		Result: struct {
			Detections []AdsDetection `json:"detections"`
		}{

			Detections: []AdsDetection{
				{
					Pts: [][2]int{
						{117, 61}, {233, 65}, {233, 95}, {117, 95},
					},
					Score: 0.3333602547645569,
				},
				{
					Pts: [][2]int{
						{186, 300}, {385, 300}, {385, 317}, {186, 317},
					},
					Score: 0.4217185080051422,
				},
				{
					Pts: [][2]int{
						{281, 321}, {397, 321}, {397, 336}, {281, 336},
					},
					Score: 0.3240920901298523,
				},
			},
		},
	}, nil
},
}

var ears = EvalAdsRecognitionEndpoints{EvalAdsRecognitionEP: func(ctx context.Context, request interface{}) (interface{}, error) {
	return AdsRecognitionResp{
		Code:    0,
		Message: "",
		Result: struct {
			Texts []struct {
				Pts  [][2]int `json:"pts"`
				Text string   `json:"text"`
			} `json:"texts"`
		}{
			Texts: []struct {
				Pts  [][2]int `json:"pts"`
				Text string   `json:"text"`
			}{
				{
					Pts: [][2]int{
						{117, 61}, {233, 65}, {233, 95}, {117, 95},
					},
					Text: "老军医包治",
				},
				{
					Pts: [][2]int{
						{186, 300}, {385, 300}, {385, 317}, {186, 317},
					},
					Text: "waa.n安远人家",
				},
				{
					Pts: [][2]int{
						{281, 321}, {395, 321}, {397, 336}, {281, 336},
					},
					Text: "传至Moobol.com",
				},
			},
		},
	}, nil
},
}

var eacs = EvalAdsClassifierEndpoints{EvalAdsClassifierEP: func(ctx context.Context, request interface{}) (interface{}, error) {
	return AdsClassifierResp{
		Code:    0,
		Message: "",
		Result: struct {
			Ads struct {
				Summary struct {
					Label string  `json:"label"`
					Score float32 `json:"score"`
				} `json:"summary"`
				Confidences []struct {
					Keys  []string `json:"keys"`
					Label string   `json:"label"`
					Score float32  `json:"score"`
				} `json:"confidences"`
			} `json:"ads"`
		}{
			Ads: struct {
				Summary struct {
					Label string  `json:"label"`
					Score float32 `json:"score"`
				} `json:"summary"`
				Confidences []struct {
					Keys  []string `json:"keys"`
					Label string   `json:"label"`
					Score float32  `json:"score"`
				} `json:"confidences"`
			}{
				Confidences: []struct {
					Keys  []string `json:"keys"`
					Label string   `json:"label"`
					Score float32  `json:"score"`
				}{
					{
						Keys:  []string{},
						Label: "normal",
						Score: 1,
					},
					{
						Keys:  []string{},
						Label: "normal",
						Score: 1,
					},
					{
						Keys:  []string{"com"},
						Label: "ads",
						Score: 0.5,
					},
				},
				Summary: struct {
					Label string  `json:"label"`
					Score float32 `json:"score"`
				}{
					Label: "ads",
					Score: 0.5,
				},
			},
		},
	}, nil
},
}

func TestAdsCensor(t *testing.T) {
	config := DEFAULT
	s1, _ := NewAdsService(config, eaqs, eads, ears, eacs)
	resp, err := s1.AdsCensor(context.Background(), pimage.ImageCensorReq{})

	assert.NoError(t, err)
	assert.Equal(t, string(pimage.REVIEW), string(resp.Suggestion))
	assert.Equal(t, "ads", resp.Details[0].Label)
	assert.Equal(t, pimage.REVIEW, resp.Details[0].Suggestion)
	assert.Equal(t, float32(0.5), resp.Details[0].Score)

}
