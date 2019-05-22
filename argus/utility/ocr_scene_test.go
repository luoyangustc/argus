package utility

import (
	"context"
	"encoding/json"
	"testing"
)

type mockOcrScene struct{}

func (mot mockOcrScene) SceneDetectEval(
	ctx context.Context, req _EvalOcrSceneDetectReq, env _EvalEnv,
) (ret _EvalOcrSceneDetectResp, err error) {
	str := `
		{
			"code": 0,
			"message": "",
			"result": {	
				"bboxes": [[10, 20, 50, 100, 60, 43, 70, 120]]
			}
		}
	`
	err = json.Unmarshal([]byte(str), &ret)
	return
}

func (mot mockOcrScene) SceneRecognizeEval(
	ctx context.Context, req _EvalOcrSceneRecognizeReq, env _EvalEnv,
) (ret _EvalOcrSceneRecognizeResp, err error) {
	str := `
		{
			"code": 0,
			"message": "",
			"result": {
				"texts": [
					{	
						"bboxes": [10, 20, 50, 100, 60, 43, 70, 120],
						"text": "this mock ocr text test"
					}
				]
			}
		}
	`
	err = json.Unmarshal([]byte(str), &ret)
	return
}

func TestOcrScene(t *testing.T) {
	service, ctx := getMockContext(t)
	service.iOcrScene = mockOcrScene{}
	ctx.Exec(`
		post http://argus.ava.ai/v1/ocr/scene
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
				"bboxes":$(bboxes),
				"text":$(text)
			}	
		}'
		equal $(code) 0
	`)
}
