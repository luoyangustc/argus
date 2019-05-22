package censor

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"qiniu.com/argus/utility/evals"
	"qiniu.com/argus/utility/server"
)

type mockTerrorClassifyNormal struct{}

func (mock mockTerrorClassifyNormal) Eval(
	ctx context.Context, req evals.SimpleReq, uid, utype uint32,
) (resp evals.TerrorClassifyResp, err error) {
	fa := `
		{
			"code": 0,
			"message": "",
			"result": {
				"confidences": [
					{
						"index": 1,
						"class": "normal",
						"score": 0.97
					}
				]
			}
		}
		`

	err = json.Unmarshal([]byte(fa), &resp)
	return
}

type mockTerrorDetectNormal struct{}

func (mock mockTerrorDetectNormal) Eval(
	ctx context.Context, req evals.SimpleReq, uid, utype uint32,
) (resp evals.TerrorDetectResp, err error) {
	fg := `
			{
			"code": 0,
			"message": "",
			"result": {
				"detections": [{}]
			}	
		}
		`
	err = json.Unmarshal([]byte(fg), &resp)
	return
}

type mockTerrorClassifyDelay struct{}

func (mock mockTerrorClassifyDelay) Eval(
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
	time.Sleep(100 * time.Millisecond)
	return
}

type mockTerrorDetectMulti struct{}

func (mock mockTerrorDetectMulti) Eval(
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
						"score": 0.99,
						"pts": [[225,195], [351,195], [351,389], [225,389]]
					},
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

func TestTerrorComplex(t *testing.T) {
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
		eTerrorClassify: mockTerrorClassifyDelay{},
		eTerrorPostDet:  nil,
	}
	ctx := server.NewHTContext(t, srv)
	// detail = false, detect score < terror_threshold
	ctx.Exec(`
		post http://test.com/v1/terror/complex
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

	// detail = true, detect score < terror_threshold
	ctx.Exec(`
		post http://test.com/v1/terror/complex
		auth |authstub -uid 1 -utype 4|
		json '{
			"data": {
				"uri": "http://test.image.jpg"  
			},
			"params": {
				"detail": true
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
					   "review":$(review),
					   "classes":[{
						   "class": $(class1),
						   "score": $(score1)
					   }]
			}	
		}'
		equal $(code) 0
		equal $(label) 1
		equal $(score) 0.97
		equal $(review) false
		equal $(class1) "guns"
		equal $(score1) 0.97
	`)

	// detail = true, detect score > terror_threshold
	srv.Config.TerrorThreshold = 0.15
	ctx = server.NewHTContext(t, srv)
	ctx.Exec(`
		post http://test.com/v1/terror/complex
		auth |authstub -uid 1 -utype 4|
		json '{
			"data": {
				"uri": "http://test.image.jpg"  
			},
			"params": {
				"detail": true
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
					   "review":$(review),
					   "classes":[{
							"class": $(class1),
							"score": $(score1)
						},
						{
							"class": $(class2),
							"score": $(score2)
						}]
			}	
		}'
		equal $(code) 0
		equal $(label) 1
		equal $(score) 0.97
		equal $(review) false
		equal $(class1) "guns"
		equal $(score1) 0.97
		equal $(class2) "knife"
		equal $(score2) 0.19
	`)

	srv.eTerrorClassify = mockTerrorClassifyNormal{}
	ctx = server.NewHTContext(t, srv)
	ctx.Exec(`
		post http://test.com/v1/terror/complex
		auth |authstub -uid 1 -utype 4|
		json '{
			"data": {
				"uri": "http://test.image.jpg"  
			},
			"params": {
				"detail": true
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
					   "review":$(review),
					   "classes":[{
							"class": $(class1),
							"score": $(score1)
						}]
			}	
		}'
		equal $(code) 0
		equal $(label) 1
		equal $(score) 0.19
		equal $(review) false
		equal $(class1) "knife"
		equal $(score1) 0.19
	`)

	srv.eTerrorDet = mockTerrorDetectNormal{}
	ctx = server.NewHTContext(t, srv)
	ctx.Exec(`
	post http://test.com/v1/terror/complex
	auth |authstub -uid 1 -utype 4|
	json '{
		"data": {
			"uri": "http://test.image.jpg"  
		},
		"params": {
			"detail": true
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
	equal $(label) 0
	equal $(score) 0.97
	equal $(review) false
	`)

	srv.eTerrorClassify = mockTerrorClassify{}
	ctx = server.NewHTContext(t, srv)
	ctx.Exec(`
	post http://test.com/v1/terror/complex
	auth |authstub -uid 1 -utype 4|
	json '{
		"data": {
			"uri": "http://test.image.jpg"  
		},
		"params": {
			"detail": true
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
				   "review":$(review),
				   "classes":[{
						"class": $(class1),
						"score": $(score1)
					}]
		}	
	}'
	equal $(code) 0
	equal $(label) 1
	equal $(score) 0.97
	equal $(review) false
	equal $(class1) "guns"
	equal $(score1) 0.97
	`)

	srv.eTerrorClassify = mockTerrorClassifyNormal{}
	srv.eTerrorDet = mockTerrorDetectMulti{}
	ctx = server.NewHTContext(t, srv)
	ctx.Exec(`
	post http://test.com/v1/terror/complex
	auth |authstub -uid 1 -utype 4|
	json '{
		"data": {
			"uri": "http://test.image.jpg"  
		},
		"params": {
			"detail": true
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
				   "review":$(review),
				   "classes":[{
						"class": $(class1),
						"score": $(score1)
					}]
		}	
	}'
	equal $(code) 0
	equal $(label) 1
	equal $(score) 0.99
	equal $(review) false
	equal $(class1) "knife"
	equal $(score1) 0.99
	`)
}
