package censor

import (
	"context"
	"encoding/json"
	"testing"

	"qiniu.com/argus/utility/evals"
	"qiniu.com/argus/utility/server"
)

type mockFaceDetect2 struct{}

func (f mockFaceDetect2) Eval(
	ctx context.Context, req evals.SimpleReq, uid, utype uint32,
) (resp evals.FaceDetectResp, err error) {
	fdt := `
		{
			"code": 0,
			"message": "",
			"result": {
				"detections": [
					{
						"index": 1,      
						"class": "face",
						"score": 0.9971,
						"pts": [[225,195], [351,195], [351,389], [225,389]]
					}
				]
			}	
		}
		`

	err = json.Unmarshal([]byte(fdt), &resp)
	return
}

type mockFaceFeature2 struct{}

func (mock mockFaceFeature2) Eval(
	ctx context.Context, req evals.FaceReq, uid, utype uint32,
) (bs []byte, err error) {
	return []byte("binary feature data stream"), nil
}

func TestPoliticianSearchV2(t *testing.T) {
	srv := &Service{
		ES: ES{
			eP: "p",
		},
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
		ePulp:               mockPulp{},
		eFaceDet:            mockFaceDetect2{},
		eFaceFeature:        mockFaceFeature2{},
		ePoliticianFFeature: mockFaceFeature2{},
		ePolitician: mockFaceSearch{
			Func: func(ctx context.Context, req interface{}, uid, utype uint32) (
				interface{}, error) {
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
						Class: "XXX1",
						Group: "leader1",
						Score: 0.994,
					},
				)

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
						Class: "XXX2",
						Group: "leader2",
						Score: 0.987,
					},
				)
				return resp, nil
			},
		},
	}
	ctx := server.NewHTContext(t, srv)
	ctx.Exec(`
	post http://test.com/v1/search/politician
		auth |authstub -uid 1 -utype 4|
		json '{
			"data": {
				"uri": "http://test.image_1.jpg"  
			},
			"params":{
				"limit":2
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
						"politician": [{
							"name": $(name1),
							"group": $(group1),
							"score":$(score1),
							"sample": {
								"url": "",
								"pts": null
							}
						},{
								"name": $(name2),
								"group":$(group2),
								"score":$(score2),
								"sample": {
								"url": "",
								"pts": null
							    }
						}]
						
	         		}
      			]
			}
		}'
		equal $(code) 0
		equal $(score) 0.9971
		equal $(name1) "XXX1"
		equal $(group1) "leader1"
		equal $(score1) 0.994
		equal $(name2) "XXX2"
		equal $(group2) "leader2"
		equal $(score2) 0.987
	`)
}
