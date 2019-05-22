package utility

import (
	"context"
	"encoding/json"
	"testing"
)

type mockBluedClassify struct{}

func (b mockBluedClassify) Eval(ctx context.Context, req *_BlueClassfiyReq, env *_EvalEnv) (ret *_BlueClassfiyResp, err error) {
	dc := `
		{
			"code": 0,
			"message": "",
			"result": {
				"confidences": [
					{
						"class": "nonhuman",
						"index": 1,
						"score": 0.987
					}
				]
			}
		}
	`
	err = json.Unmarshal([]byte(dc), &ret)
	return
}

type mockBluedD struct{}

func (b mockBluedD) Eval(ctx context.Context, req *_BlueDReq, env *_EvalEnv) (ret *_BlueDResp, err error) {

	dd := `
	{
		"code": 0,
		"message": "",
		"result": {
			"detections": [
				{
					"area_ratio": 0.2712,
					"class": "tattoo",
					"index": 8,
					"score": 0.9973000288009644,
					"pts": [
						[
							269,
							14
						],
						[
							499,
							14
						],
						[
							499,
							312
						],
						[
							269,
							312
						]
					]
				},
				{
					"area_ratio": 0.25683,
					"class": "tattoo",
					"index": 8,
					"score": 0.967,
					"pts": [
						[
							25,
							29
						],
						[
							246,
							29
						],
						[
							246,
							322
						],
						[
							25,
							322
						]
					]
				}
			]
		}
	}
	`
	err = json.Unmarshal([]byte(dd), &ret)
	return
}
func TestBluedDetection(t *testing.T) {
	service, ctx := getMockContext(t)
	service.iBluedD = mockBluedD{}
	service.iBluedClassify = mockBluedClassify{}
	ctx.Exec(`
		post http://argus.ava.ai/v1/blued/detection
		auth |authstub -uid 1 -utype 4|
		json '{
			"data": {
				"uri": "http://77g4l9.com5.z0.glb.qiniucdn.com/ingfiles/008/441/530/8441530_19116_1510330373.png"  
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
						"area_ratio":$(ratio)
					},
					{
					   "class":$(class),
					   "index":$(index),
					   "score":$(score)
					}
				]
			}	
		}'
		equal $(code) 0
		equal $(ratio) 0.2712
		equal $(class) "tattoo"
		equal $(index) 8
		equal $(score) 0.967

	`)
}
