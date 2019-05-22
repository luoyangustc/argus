package utility

import (
	"context"
	"encoding/json"
	"testing"
)

type mockDetect struct {
}

func (o mockDetect) Eval(ctx context.Context, args _EvalDetectionReq, env _EvalEnv) (ret _EvalDetectionResp, err error) {
	str := `
		{
			"code": 0,
			"message": "",
			"result": {
				"detections": [
					{
						"class": "dog",
						"index": 58,
						"pts": [[138,200],[305,200],[305,535],[138,535]],
						"score": 0.98
					}
				]
			}
	}
	`
	err = json.Unmarshal([]byte(str), &ret)
	return
}

func TestDetection(t *testing.T) {
	service, ctx := getMockContext(t)
	service.iDetection = mockDetect{}
	ctx.Exec(`
		post http://argus.ava.ai/v1/detect
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
				"detections": [
					{
					   "class":$(class),
					   "index":$(index),
					   "score":$(score)
					}
				]
			}	
		}'
		equal $(code) 0
		equal $(class) dog
		equal $(score) 0.98
	`)
}
