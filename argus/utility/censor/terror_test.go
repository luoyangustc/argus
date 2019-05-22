package censor

import (
	"context"
	"encoding/json"
	"testing"

	"qiniu.com/argus/utility/evals"
	"qiniu.com/argus/utility/server"
)

type mockTerrorClassify struct{}

func (mock mockTerrorClassify) Eval(
	ctx context.Context, req evals.SimpleReq, uid, utype uint32,
) (resp evals.TerrorClassifyResp, err error) {
	fa := `
		{
			"code": 0,
			"message": "",
			"result": {
				"confidences": [
					{
						"index": 47,
						"class": "guns",
						"score": 0.97
					}
				]
			}
		}
		`

	err = json.Unmarshal([]byte(fa), &resp)
	return
}

type mockTerrorDetect struct{}

func (mock mockTerrorDetect) Eval(
	ctx context.Context, req evals.SimpleReq, uid, utype uint32,
) (resp evals.TerrorDetectResp, err error) {
	fg := `
			{
			"code": 0,
			"message": "",
			"result": {
				"detections": [
					{
						"index": 1,      
						"class": "knife",
						"score": 0.19,
						"pts": [[225,195], [351,195], [351,389], [225,389]]
					}
				]
			}	
		}
		`
	err = json.Unmarshal([]byte(fg), &resp)
	return
}

type mockTerrorPreDetect struct{}

func (mock mockTerrorPreDetect) Eval(
	ctx context.Context, req evals.SimpleReq, uid, utype uint32,
) (resp evals.TerrorPreDetectResp, err error) {
	fg := `
			{
			"code": 0,
			"message": "",
			"result": {
				"detections": [
					{
						"index": 1,      
						"score": 0.19,
						"pts": [[225,195], [351,195], [351,389], [225,389]]
					}
				]
			}	
		}
		`
	err = json.Unmarshal([]byte(fg), &resp)
	return
}

func TestTerror(t *testing.T) {
	srv := &Service{
		ES: ES{
			eTDP: "dp", eTD: "d", eTC: "c", eTDPo: "dpo",
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
		eTerrorPreDet:   mockTerrorPreDetect{},
		eTerrorDet:      mockTerrorDetect{},
		eTerrorClassify: mockTerrorClassify{},
		eTerrorPostDet:  nil,
	}
	ctx := server.NewHTContext(t, srv)
	ctx.Exec(`
		post http://test.com/v1/terror
		auth |authstub -uid 1 -utype 4|
		json '{
			"data": {
				"uri": "http://test.image.jpg"  
			}   
		}'
		ret 200
		header Content-Type $(mime) 
		equal $(mime) 'application/json'
		echo $(resp.body)
		json '{
			"code": $(code),
			"result":{
					   "label":$(label),
					   "score":$(score),
					   "review":$(review)
			}	
		}'
		equal $(code) 0
		equal $(label) 1
		equal $(score) 0.97
		equal $(review) false

	`)
}
