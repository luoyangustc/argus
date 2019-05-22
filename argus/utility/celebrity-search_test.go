package utility

 import (
 	"context"
// 	"testing"
 )

// type mockCelebrityManager struct {
// }

// func (mi mockCelebrityManager) AddCelebrity(context.Context, ...Celebrity) error { return nil }
// func (mi mockCelebrityManager) Celebritys(context.Context) ([]string, error)     { return nil, nil }
// func (mi mockCelebrityManager) Delete(context.Context, string, string) error     { return nil }
// func (mi mockCelebrityManager) CelebrityImages(context.Context, ...string) ([]string, error) {
// 	return nil, nil
// }
// func (mi mockCelebrityManager) All(context.Context) ([]Celebrity, string, error) {
// 	return []Celebrity{
// 		{Name: "Jack.Ma", Url: "xxx1", ID: "xxx11", Feature: []byte("xxxxx")},
// 		{Name: "Roby", Url: "xxx2", ID: "xxx22", Feature: []byte("xxxxx")},
// 		{Name: "Macxs", Url: "xxx3", ID: "xxx33", Feature: []byte("xxxxx")},
// 		{Name: "Msocre", Url: "xxx4", ID: "xxx44", Feature: []byte("xxxxx")},
// 		{Name: "Frank", Url: "xxx5", ID: "xxx55", Feature: []byte("xxxxx")},
// 	}, "", nil
// }

type mockFaceFeatureV2 struct {
}

func (mockFaceFeatureV2) Eval(
	ctx context.Context, req _EvalFaceReq, env _EvalEnv,
) (bs []byte, err error) {
	return []byte("binary feature data"), nil
}

// func TestCelebritySearch(t *testing.T) {
// 	service, ctx := getMockContext(t)
// 	service.iFaceSearchCelebrity = &mockFaceSearch2{
// 		Func: func(ctx context.Context, req _EvalFaceSearchReq, env _EvalEnv) (resp _EvalFaceSearchResp, err error) {
// 			resp.Result.Class = "XXX"
// 			resp.Result.Score = 0.998
// 			return
// 		},
// 	}
// 	service._CelebrityManager = mockCelebrityManager{}
// 	service.iFacexSearch = mockDynamicFacexSearch{}
// 	service.iFaceDetect = mockFaceDetect{}
// 	service.iFaceFeatureV2 = mockFaceFeatureV2{}

// 	ctx.Exec(`
// 		post http://argus.ava.ai/v1/celebrity/search
// 		auth |authstub -uid 1 -utype 4|
// 		json '{
// 			"data": {
// 				"uri": "http://test.image.jpg"
// 			}
// 		}'
// 		ret 200
// 		header Content-Type $(mime)
// 		equal $(mime) 'application/json'
// 		echo $(resp.body)
// 		json '{
// 			"code": $(code),
// 			"message": "",
// 			"result": {
// 				"detections": [
// 					{
// 						"value": {
// 							"name": $(name),
// 							"score":$(score),
// 							"review": $(review)
// 						}
// 					}
// 				]
// 			}
// 		}'
// 		equal $(code) 0
// 		equal $(name) XXX
// 		equal $(score) 0.998
// 		equal $(review) false

// 	`)
// }
