package censor

import (
	"context"
	"testing"

	"qiniu.com/argus/utility/server"
)

func TestPulpV0(t *testing.T) {
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
		ePulp: mockPulp{},
	}
	ctx := server.NewHTContext(t, srv)

	ctx.Exec(`
		post http://test.com/v1/pulp/recognition
		auth |authstub -uid 1 -utype 4|
		json '{
			"image": [
				"http://test.image1.jpg"  
			] 
		}'
		ret 200
		header Content-Type $(mime) 
		equal $(mime) 'application/json'
		echo $(resp.body)
		json '{
			"code": $(code),
			"pulp":{
				"fileList":[
					{
					  "result":{
					   		"label":$(label),
					   		"rate":$(score),
							"name": $(file),
					   		"review":$(review)
					   }
					}
				]
			}	
		}'
		equal $(code) 0
		equal $(label) 2
		equal $(score) 0.40449440
		equal $(review) true
		equal $(file) "http://test.image1.jpg"
	`)
}
