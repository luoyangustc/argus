package censor

import (
	"context"
	"testing"

	"qiniu.com/argus/utility/evals"

	"qiniu.com/argus/service/service/image/ads"
	"qiniu.com/argus/service/service/image/politician"
	"qiniu.com/argus/service/service/image/pulp"
	"qiniu.com/argus/service/service/image/terror"

	"github.com/stretchr/testify/assert"
)

var eaqs = ads.EvalAdsQrcodeEndpoints{EvalAdsQrcodeEP: func(ctx context.Context, request interface{}) (interface{}, error) {
	return ads.AdsQrcodeResp{
		Code:    0,
		Message: "",
		Result: struct {
			Detections []ads.QrcodeDetection `json:"detections"`
		}{
			Detections: []ads.QrcodeDetection{
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

var eads = ads.EvalAdsDetectEndpoints{EvalAdsDetectEP: func(ctx context.Context, request interface{}) (interface{}, error) {
	return ads.AdsDetectionResp{
		Code:    0,
		Message: "",
		Result: struct {
			Detections []ads.AdsDetection `json:"detections"`
		}{

			Detections: []ads.AdsDetection{
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

var ears = ads.EvalAdsRecognitionEndpoints{EvalAdsRecognitionEP: func(ctx context.Context, request interface{}) (interface{}, error) {
	return ads.AdsRecognitionResp{
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

var eacs = ads.EvalAdsClassifierEndpoints{EvalAdsClassifierEP: func(ctx context.Context, request interface{}) (interface{}, error) {
	return ads.AdsClassifierResp{
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

var efds = politician.EvalFaceDetectEndpoints{EvalFaceDetectEP: func(ctx context.Context, request interface{}) (interface{}, error) {
	resp := evals.FaceDetectResp{}
	resp.Result.Detections = []evals.FaceDetection{
		evals.FaceDetection{
			Index: 1,
			Class: "face",
			Score: 0.9971,
			Pts:   [][2]int{{225, 195}, {351, 195}, {351, 389}, {225, 389}},
		},
	}
	return resp, nil
}}

var effs = politician.EvalFaceFeatureEndpoints{EvalFaceFeatureEP: func(ctx context.Context, request interface{}) (interface{}, error) {
	return []byte("feature string"), nil
}}

var eps = politician.EvalPoliticianEndpoints{EvalPoliticianEP: func(ctx context.Context, request interface{}) (interface{}, error) {
	var resp evals.FaceSearchRespV2
	resp.Result.Confidences = append(
		resp.Result.Confidences, struct {
			Index  int     `json:"index"`
			Class  string  `json:"class"`
			Group  string  `json:"group"`
			Score  float32 `json:"score"`
			Sample struct {
				URL string   `json:"url"`
				Pts [][2]int `json:"pts"`
				ID  string   `json:"id"`
			} `json:"sample"`
		}{
			Class: "XXX",
			Group: "affairs_official_gov",
			Score: 0.998,
		},
	)
	return resp, nil
}}

var epps = pulp.EvalPulpEndpoints{EvalPulpEP: func(ctx context.Context, request interface{}) (interface{}, error) {
	return evals.PulpResp{
		Code:    0,
		Message: "",
		Result: struct {
			Checkpoint  string
			Confidences []struct {
				Index int     `json:"index"`
				Class string  `json:"class"`
				Score float32 `json:"score"`
			} `json:"confidences"`
		}{
			Confidences: []struct {
				Index int     `json:"index"`
				Class string  `json:"class"`
				Score float32 `json:"score"`
			}{
				{
					Class: "normal",
					Index: 2,
					Score: 0.6,
				},
				{
					Class: "sexy",
					Index: 1,
					Score: 0.5,
				},
				{
					Class: "pulp",
					Index: 0,
					Score: 0.4,
				},
			},
		},
	}, nil
}}

var epfs = pulp.EvalPulpFilterEndpoints{EvalPulpFilterEP: func(ctx context.Context, request interface{}) (interface{}, error) {
	return evals.PulpResp{
		Code:    0,
		Message: "",
		Result: struct {
			Checkpoint  string
			Confidences []struct {
				Index int     `json:"index"`
				Class string  `json:"class"`
				Score float32 `json:"score"`
			} `json:"confidences"`
		}{
			Checkpoint: "endpoint",
			Confidences: []struct {
				Index int     `json:"index"`
				Class string  `json:"class"`
				Score float32 `json:"score"`
			}{
				{
					Class: "normal",
					Index: 2,
					Score: 0.6,
				},
				{
					Class: "sexy",
					Index: 1,
					Score: 0.5,
				},
				{
					Class: "pulp",
					Index: 0,
					Score: 0.4,
				},
			},
		},
	}, nil
}}

var etms = terror.EvalTerrorMixupEndpoints{EvalTerrorMixupEP: func(ctx context.Context, request interface{}) (interface{}, error) {
	return terror.TerrorMixupResp{
		Code:    0,
		Message: "",
		Result: struct {
			Checkpoint  string `json:"checkpoint"`
			Confidences []struct {
				Index int     `json:"index"`
				Class string  `json:"class"`
				Score float32 `json:"score"`
			} `json:"confidences"`
		}{
			Confidences: []struct {
				Index int     `json:"index"`
				Class string  `json:"class"`
				Score float32 `json:"score"`
			}{
				{
					Index: 47,
					Class: "guns",
					Score: 0.45,
				},
				{
					Index: 40,
					Class: "terrorist",
					Score: 0.55,
				},
				{
					Index: 45,
					Class: "knives",
					Score: 0.99,
				},
			},
			Checkpoint: "terror-detect",
		},
	}, nil
}}

var etds = terror.EvalTerrorDetectEndpoints{EvalTerrorDetectEP: func(ctx context.Context, request interface{}) (interface{}, error) {
	return evals.TerrorDetectResp{
		Code:    0,
		Message: "",
		Result: struct {
			Detections []evals.TerrorDetection `json:"detections"`
		}{
			Detections: []evals.TerrorDetection{
				{
					Index: 1,
					Class: "knife",
					Score: 0.19,
					Pts:   [][2]int{{225, 195}, {351, 195}, {351, 389}, {225, 389}},
				},
			},
		},
	}, nil
}}

func TestCensor(t *testing.T) {
	s, _ := NewCensorService(DEFAULT, epps, epfs, efds, effs, eps, etms, etds, eaqs, eads, ears, eacs)
	resp, err := s.Censor(context.Background(), ImageCensorReq{
		Params: struct {
			Type   []string `json:"type,omitempty"`
			Detail bool     `json:"detail"`
		}{

			Type:   []string{"pulp", "terror", "politician"},
			Detail: true,
		},
	})

	assert.Nil(t, err)
	assert.Equal(t, 1, resp.Result.Label)
	assert.Equal(t, float32(0.998), resp.Result.Score)
	for _, dt := range resp.Result.Details {
		if dt.Type == "politician" || dt.Type == "terror" {
			assert.Equal(t, 1, dt.Label)
		}
		if dt.Type == "terror" {
			assert.Equal(t, "knife", dt.Class)
		}
		if dt.Type == "pulp" {
			assert.Equal(t, 2, dt.Label)
		}
	}
}
