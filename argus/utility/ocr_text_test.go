package utility

import (
	"context"
	"encoding/json"
	"testing"
)

type mockOcrText struct{}

func (mot mockOcrText) ClassifyEval(
	ctx context.Context, req _EvalOcrTextClassifyReq, env _EvalEnv,
) (ret _EvalOcrTextClassifyResp, err error) {
	str := `
		{
			"code": 0,
			"message": "",
			"result": {
				"confidences": [
					{
						"index": 31,
						"class": "weibo",
						"score": 0.9999
					}
				]
			}
		}
	`
	err = json.Unmarshal([]byte(str), &ret)
	return
}

func (mot mockOcrText) DetectEval(
	ctx context.Context, req _EvalOcrCtpnReq, env _EvalEnv,
) (ret _EvalOcrCtpnResp, err error) {
	str := `
		{
			"code": 0,
			"message": "",
			"result": {	
				"bboxes": [[[10,20],[50,20],[50,100],[10,100]],[[60,43],[70,43],[70,120],[60,120]]],
				"image_type": "weibo",
				"area_ratio": 0.6325
			}
		}
	`
	err = json.Unmarshal([]byte(str), &ret)
	return
}

func (mot mockOcrText) RecognizeEval(
	ctx context.Context, req _EvalOcrTextRecognizeReq, env _EvalEnv,
) (ret _EvalOcrTextRecognizeResp, err error) {
	str := `
		{
			"code": 0,
			"message": "",
			"result": {	
				"bboxes": [[[10,20],[50,20],[50,100],[10,100]],[[60,43],[70,43],[70,120],[60,120]]],
				"texts": ["this mock ocr text test","this mock ocr text test, again"]
			}
		}
	`
	err = json.Unmarshal([]byte(str), &ret)
	return
}

func (mot mockOcrText) SceneDetectEval(
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

func (mot mockOcrText) SceneRecognizeEval(
	ctx context.Context, req _EvalOcrSceneRecognizeReq, env _EvalEnv,
) (ret _EvalOcrSceneRecognizeResp, err error) {
	str := `
		{
			"code": 0,
			"message": "",
			"result": {
				texts: [
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

func TestOcrText(t *testing.T) {
	service, ctx := getMockContext(t)
	service.iOcrText = mockOcrText{}
	ctx.Exec(`
		post http://argus.ava.ai/v1/ocr/text
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
				"type":$(type),
				"bboxes":$(bboxes),
				"texts":$(texts)
			}	
		}'
		equal $(code) 0
		equal $(type) weibo
		equal $(bboxes) '[[[10,20],[50,20],[50,100],[10,100]],[[60,43],[70,43],[70,120],[60,120]]]'
		equal $(texts) '["this mock ocr text test","this mock ocr text test, again"]'
	`)
}
