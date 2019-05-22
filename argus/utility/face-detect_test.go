package utility

import (
	"context"
	"encoding/json"
	"testing"

	"qiniu.com/argus/utility/evals"
	"qiniu.com/argus/utility/server"
)

type mockStaticFaceDetect struct {
	Resp evals.FaceDetectResp
	Err  error
}

func (mock mockStaticFaceDetect) Eval(context.Context, evals.SimpleReq, uint32, uint32) (evals.FaceDetectResp, error) {
	return mock.Resp, mock.Err
}

type mockFaceAge struct{}

func (mock mockFaceAge) Eval(ctx context.Context, req _EvalFaceAgeReq, env _EvalEnv) (resp _EvalFaceResp, err error) {
	fa := `
		{
			"code": 0,
			"message": "",
			"result": {
				"confidences": [
					{
						"index": 0,
						"class": "2-4",
						"score": 0.97
					}
				]
			}
		}
		`

	err = json.Unmarshal([]byte(fa), &resp)
	return
}

type mockFaceGender struct{}

func (mock mockFaceGender) Eval(
	ctx context.Context, req _EvalFaceReq, env _EvalEnv,
) (resp _EvalFaceResp, err error) {
	fg := `
			{
				"code": 0,
				"message": "",
				"result": {
					"confidences": [
						{
							"index": 0,
							"class": "Male",
							"score": 0.8863510489463806
						}
					]
				}
			}
		`
	err = json.Unmarshal([]byte(fg), &resp)
	return
}

func TestFaceDetect(t *testing.T) {

	srv := (&FaceDetectSrv{eFD: "fd", eFA: "fa", eFG: "fg"}).
		Init(nil, server.MockStaticServer{
			GetEvalF: func(name string) interface{} {
				switch name {
				case "fd":
					resp := evals.FaceDetectResp{}
					resp.Result.Detections = []evals.FaceDetection{
						evals.FaceDetection{
							Index: 1,
							Class: "face",
							Score: 0.9971,
							Pts:   [][2]int{{225, 195}, {351, 195}, {351, 389}, {225, 389}},
						},
					}
					return mockStaticFaceDetect{Resp: resp}
				case "fa":
					return mockFaceAge{}
				case "fg":
					return mockFaceGender{}
				default:
					return nil
				}
			},
		})

	ctx := server.NewHTContext(t, srv)
	ctx.Exec(`
		post http://test.com/v1/face/detect
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
				"detections": [
					 {
						"age":{
							"value":$(agev)
						},
						"gender":{
							"value":$(genderv)
						}
					}
				]	
			}
		}'
		equal $(code) 0
		equal $(agev) 2.91
		equal $(genderv) "Male"

	`)
}
