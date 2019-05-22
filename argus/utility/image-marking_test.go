package utility

import (
	"context"
	"encoding/json"
	"testing"
)

type mockObjectClassify struct {
}

func (o mockObjectClassify) Eval(ctx context.Context, args _EvalObjectClassifyReq, env _EvalEnv) (ret _EvalObjectClassifyResp, err error) {
	str := `
		{
			"code": 0,
			"message": "",
			"result": {
				"confidences": [
					{
						"class": "bloodhound, sleuthhound",
						"index": 332,
						"score": 0.54565
					}
				]
			}
	}
	`
	err = json.Unmarshal([]byte(str), &ret)
	return
}

func TestImageMarking(t *testing.T) {
	service, ctx := getMockContext(t)
	service.iScene = mockScene{}
	service.iDetection = mockDetect{}
	service.iObjectClassify = mockObjectClassify{}
	ctx.Exec(`
		post http://argus.ava.ai/v1/image/label
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
				"confidences": [
					{
					   "class":$(class1),
					   "score":$(score1)
					},
					{
						"class":$(class2),
						"score":$(score2)
					 },
					 {
						"class":$(class3),
						"score":$(score3)
					 }
				]
			}	
		}'
		equal $(code) 0
		equal $(class1) "dog"
		equal $(score1) 0.8377719
		equal $(class2) "bloodhound, sleuthhound"
		equal $(score2) 0.81715554
		equal $(class3) "/v/valley"
		equal $(score3) 0.6741599
	`)
}
