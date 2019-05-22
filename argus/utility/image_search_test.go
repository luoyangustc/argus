package utility

import (
	"context"
	"testing"
)

type mockImageSearch struct {
	Func func(context.Context, string, _EvalImageSearchReq, _EvalEnv) (_EvalImageSearchResp, error)
}

func (mock mockImageSearch) Eval(
	ctx context.Context, name string, req _EvalImageSearchReq, env _EvalEnv,
) (_EvalImageSearchResp, error) {
	return mock.Func(ctx, name, req, env)
}

func TestImageSearch(t *testing.T) {
	service, ctx := getMockContext(t)
	service.iFeatureV2 = mockFeature{}
	service.iImageSearch = mockImageSearch{
		Func: func(
			ctx context.Context,
			name string,
			req _EvalImageSearchReq,
			env _EvalEnv,
		) (resp _EvalImageSearchResp, err error) {

			resp.Code = 0
			resp.Result = []struct {
				Class string  `json:"class"`
				Label string  `json:"label"`
				Score float32 `json:"score"`
			}{
				struct {
					Class string  `json:"class"`
					Label string  `json:"label"`
					Score float32 `json:"score"`
				}{
					Class: "XXX",
					Score: 0.998,
				},
			}
			return
		},
	}
	ctx.Exec(`
	post http://argus.ava.ai/v1/image/search/xx
		auth |authstub -uid 1 -utype 4|
		json '{
			"data": {
				"uri": "http://test.image_1.jpg"  
			}
		}'
		ret 200
		header Content-Type $(mime) 
		equal $(mime) 'application/json'
		echo $(resp.body)
		json '{
			"code": $(code),
			"result":[
            	{
					"url": $(url),
					"score":$(score)
         		}
    		]
		}'
		equal $(code) 0
		equal $(url) "XXX"
		equal $(score) 0.998
	`)
}
