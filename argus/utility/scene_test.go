package utility

import (
	"context"
	"encoding/json"
	"testing"
)

type mockScene struct {
}

func (o mockScene) Eval(ctx context.Context, args _EvalSceneReq, env _EvalEnv) (ret _EvalSceneResp, err error) {
	str := `
		{
			"code": 0,
			"message": "",
			"result": {
				"confidences": [
					{
						"class": "/v/valley",
						"index": 345,
						"label": ["outdoor","landscape"],
						"score": 0.3064
					}
				]
			}
	}
	`
	err = json.Unmarshal([]byte(str), &ret)
	return
}

func TestScene(t *testing.T) {
	service, ctx := getMockContext(t)
	service.iScene = mockScene{}
	ctx.Exec(`
		post http://argus.ava.ai/v1/scene
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
					   "class":$(class),
					   "index":$(index),
					   "score":$(score)
					}
				]
			}	
		}'
		equal $(code) 0
		equal $(class) /v/valley
		equal $(score) 0.3064
	`)
}
