package utility

import (
	"context"
	"testing"

	"qiniu.com/argus/utility/evals"
	"qiniu.com/argus/utility/server"
)

type mockStaticFaceFeature struct {
	BS  []byte
	Err error
}

func (mock mockStaticFaceFeature) Eval(
	ctx context.Context, req evals.FaceReq, uid, utype uint32,
) (bs []byte, err error) {
	return mock.BS, mock.Err
}

func TestFaceSim(t *testing.T) {

	srv := (&FaceSimSrv{eFD: "fd", eFF: "ff"}).
		Init([]byte(`{"threshold": 0.5}`), server.MockStaticServer{
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
				case "ff":
					return mockStaticFaceFeature{BS: make([]byte, 8)}
				default:
					return nil
				}
			},
		})

	ctx := server.NewHTContext(t, srv)
	ctx.Exec(`
	post http://test.com/v1/face/sim
		auth |authstub -uid 1 -utype 4|
		json '{
			"data": [
				{
				"uri": "http://test.image_1.jpg"  
			   },{
				"uri": "http://test.image_1.jpg"
			   }]
		}'
		ret 200
		header Content-Type $(mime) 
		equal $(mime) 'application/json'
		echo $(resp.body)
		json '{
			"code": $(code),
			"result":{
				"faces": [{
		               		 "score":$(score)
	           			},{}],
				"similarity":$(sim),
				"same":$(sam)
			}
		}'
		equal $(code) 0
		equal $(score) 0.9971
		equal $(sim) 0
		equal $(sam) false
	`)
}
