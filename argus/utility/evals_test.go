package utility

import (
	"context"
	"encoding/json"

	"qiniu.com/argus/utility/evals"
)

type mockFaceDetect struct{}

func (f mockFaceDetect) Eval(
	ctx context.Context, req _EvalFaceDetectReq, env _EvalEnv,
) (resp _EvalFaceDetectResp, err error) {
	fdt := `
		{
			"code": 0,
			"message": "",
			"result": {
				"detections": [
					{
						"index": 1,      
						"class": "face",
						"score": 0.9971,
						"pts": [[225,195], [351,195], [351,389], [225,389]]
					}
				]
			}	
		}
		`

	err = json.Unmarshal([]byte(fdt), &resp)
	return
}

type mockFeature struct{}

func (mock mockFeature) Eval(
	ctx context.Context, req _EvalImageReq, env _EvalEnv,
) (bs []byte, err error) {
	return []byte("binary feature data stream"), nil
}

//----------------------------------------------------------------------------//

type mockFaceDetect2 struct{}

func (f mockFaceDetect2) Eval(
	ctx context.Context, req evals.SimpleReq, uid, utype uint32,
) (resp evals.FaceDetectResp, err error) {
	fdt := `
		{
			"code": 0,
			"message": "",
			"result": {
				"detections": [
					{
						"index": 1,      
						"class": "face",
						"score": 0.9971,
						"pts": [[225,195], [351,195], [351,389], [225,389]]
					}
				]
			}	
		}
		`

	err = json.Unmarshal([]byte(fdt), &resp)
	return
}

type mockFaceFeature2 struct{}

func (mock mockFaceFeature2) Eval(
	ctx context.Context, req evals.FaceReq, uid, utype uint32,
) (bs []byte, err error) {
	return []byte("binary feature data stream"), nil
}
