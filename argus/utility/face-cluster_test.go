package utility

import (
	"context"
	"testing"
)

type mockFaceFeature struct{}

func (mock mockFaceFeature) Eval(
	ctx context.Context, req _EvalFaceReq, env _EvalEnv,
) (bs []byte, err error) {
	return []byte("binary feature data stream"), nil
}

type mockFaceCluster struct{}

func (mock mockFaceCluster) Eval(
	ctx context.Context, req _EvalFaceClusterReq, env _EvalEnv,
) (resp _EvalFaceClusterResp, err error) {
	for _ = range req.Data {
		resp.Result.Fcluster = append(
			resp.Result.Fcluster,
			_EvalFaceClusterDetail{
				ID:         2,
				CenterDist: 0.9876,
			},
		)
	}
	return
}

func TestFaceCluster(t *testing.T) {
	service, ctx := getMockContext(t)
	service.iFaceDetect = mockFaceDetect{}
	service.facexFeatureV3 = mockStaticFaceFeature{BS: []byte{}}
	service.iFaceCluster = mockFaceCluster{}
	ctx.Exec(`
	post http://argus.ava.ai/v1/face/cluster
		auth |authstub -uid 1 -utype 4|
		json '{
			"data": [
				{
				"uri": "http://test.image_1.jpg"  
				},
				{
				"uri": "http://test.image_2.jpg"  
				},
				{
				"uri": "http://test.image_3.jpg"  
				}
			]   
		}'
		ret 200
		header Content-Type $(mime) 
		equal $(mime) 'application/json'
		echo $(resp.body)
		json '{
			"code": $(code),
			"result":{
				 "cluster": [
            		[
                		{
                   			"boundingBox":{
                       		 "score":$(score)
                   			}
                 		}
            		],
            		[
                 		{
                    		"group":{
                       			"id": $(id)
                  			}
               		 	}
            		],
					[
                 		{
                    		"group":{
                       			"center_dist": $(dist)
                  			}
               		 	}
            		]  
      			]
			}
		}'
		equal $(code) 0
		equal $(score) 0.9971
		equal $(id) 2
		equal $(dist) 0.9876
	`)
}
