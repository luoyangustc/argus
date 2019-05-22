package censor

import (
	"context"
	"testing"

	"qiniu.com/argus/utility/evals"
	"qiniu.com/argus/utility/server"
)

var _ evals.IFaceDetect = mockFaceDetect{}

type mockFaceDetect struct {
	Func func(context.Context, evals.SimpleReq, uint32, uint32) (evals.FaceDetectResp, error)
}

func (mock mockFaceDetect) Eval(
	ctx context.Context, req evals.SimpleReq, uid, utype uint32,
) (evals.FaceDetectResp, error) {
	return mock.Func(ctx, req, uid, utype)
}

var _ evals.IFaceFeature = mockFaceFeature{}

type mockFaceFeature struct {
	Func func(context.Context, evals.FaceReq, uint32, uint32) ([]byte, error)
}

func (mock mockFaceFeature) Eval(
	ctx context.Context, req evals.FaceReq, uid, utype uint32,
) ([]byte, error) {
	return mock.Func(ctx, req, uid, utype)
}

var _ evals.IPolitician = mockFaceSearch{}

type mockFaceSearch struct {
	Func func(context.Context, interface{}, uint32, uint32) (interface{}, error)
}

func (mock mockFaceSearch) Eval(
	ctx context.Context, req interface{}, uid, utype uint32,
) (interface{}, error) {
	return mock.Func(ctx, req, uid, utype)
}

func TestPoliticianSearch(t *testing.T) {
	srv := &Service{ES: ES{eFD: "fd", eFF: "ff", ePO: "po"},
		Config: Config{
			TerrorThreshold:     0.25,
			PoliticianThreshold: []float32{0.6, 0.66, 0.72},
			PulpReviewThreshold: 0.89,
		},
		IServer: server.MockStaticServer{
			ParseImageF: func(ctx context.Context, uri string) (server.Image, error) {
				return mockImageParse{}.ParseImage(ctx, uri)
			},
		},
		eFaceDet: mockFaceDetect{
			Func: func(
				ctx context.Context, req evals.SimpleReq, uid, utype uint32,
			) (evals.FaceDetectResp, error) {
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
			},
		},
		eFaceFeature: mockFaceFeature{
			Func: func(
				tx context.Context, req evals.FaceReq, uid, utype uint32,
			) ([]byte, error) {
				return make([]byte, 8), nil
			},
		},
		ePoliticianFFeature: mockFaceFeature{
			Func: func(
				tx context.Context, req evals.FaceReq, uid, utype uint32,
			) ([]byte, error) {
				return make([]byte, 8), nil
			},
		},
		ePolitician: mockFaceSearch{
			Func: func(
				ctx context.Context, req interface{}, uid, utype uint32,
			) (interface{}, error) {
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
						Group: "governer",
						Score: 0.998,
					},
				)
				return resp, nil
			},
		},
	}
	ctx := server.NewHTContext(t, srv)
	ctx.Exec(`
	post http://test.com/v1/face/search/politician
		auth |authstub -uid 1 -utype 4|
		json '{
			"data": {
				"uri": "http://test.image_1.jpg"  
			}
		}'
		ret 200
		header Content-Type $(mime) 
		equal $(mime) 'application/json'
		echo $(resp.body)
		json '{
			"code": $(code),
			"result":{
				"detections": [
              		{
	        			"boundingBox":{
		               		 "score":$(score)
	           			},
						"value": {
							"name": $(name),
							"group": $(group),
							"score":$(score2)
						},
						"sample": {
							"url": "",
							"pts": null
						}
	         		}
      			]
			}
		}'
		equal $(code) 0
		equal $(score) 0.9971
		equal $(name) "XXX"
		equal $(group) "governer"
		equal $(score2) 0.998
	`)
}
