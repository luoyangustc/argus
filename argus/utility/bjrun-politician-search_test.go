package utility

import (
	"context"
	"encoding/json"
	"testing"

	"qiniu.com/argus/utility/evals"
)

type mockDynamicFacexSearch struct {
}

func (fm mockDynamicFacexSearch) Eval(
	ctx context.Context, req _EvalFacexSearchReq, env _EvalEnv,
) (_EvalFacexSearchResp, error) {
	str := `
	{
		"code":0,
        "message":"",
        "result":{
            "index":0,
            "score":0.987
        }
	}	 
	`
	var resp _EvalFacexSearchResp
	err := json.Unmarshal([]byte(str), &resp)
	return resp, err
}

type mockBjrunImageSearch struct {
}

func (bm mockBjrunImageSearch) Eval(context.Context, _EvalBjrunImageSearchReq, _EvalEnv) (_EvalBjrunImageSearchResp, error) {
	str := `
	 {
		"code":0,
		"message":"",
		"result":[
			{
				"index":0,
				"score":0.768
			},
			{
				"index":1,
				"score":0.876
			},
			{
				"index":2,
				"score":0.755
			}
		]
	}
	`
	var resp _EvalBjrunImageSearchResp
	err := json.Unmarshal([]byte(str), &resp)
	return resp, err
}

type mockFaceSearch struct {
	Func func(context.Context, interface{}, uint32, uint32) (interface{}, error)
}

func (mock mockFaceSearch) Eval(
	ctx context.Context, req interface{}, uid, utype uint32,
) (interface{}, error) {
	return mock.Func(ctx, req, uid, utype)
}

type mockImageManager struct {
}

func (mi mockImageManager) AddImage(context.Context, ...Image) error                 { return nil }
func (mi mockImageManager) Labels(context.Context) ([]string, error)                 { return nil, nil }
func (mi mockImageManager) LabelImages(context.Context, ...string) ([]string, error) { return nil, nil }
func (mi mockImageManager) Delete(context.Context, string, string) error             { return nil }
func (mi mockImageManager) All(context.Context) ([]Image, string, error) {
	return []Image{
		{Label: "landscore", Url: "xxx1", ID: "xxx11", Feature: []byte("xxxxx")},
		{Label: "man", Url: "xxx2", ID: "xxx22", Feature: []byte("xxxxx")},
		{Label: "house", Url: "xxx3", ID: "xxx33", Feature: []byte("xxxxx")},
		{Label: "animal", Url: "xxx4", ID: "xxx44", Feature: []byte("xxxxx")},
		{Label: "picture", Url: "xxx5", ID: "xxx55", Feature: []byte("xxxxx")},
	}, "", nil
}

type mockPoliticianManager struct {
}

func (mi mockPoliticianManager) AddPolitician(context.Context, ...Politician) error { return nil }
func (mi mockPoliticianManager) Politicians(context.Context) ([]string, error)      { return nil, nil }
func (mi mockPoliticianManager) Delete(context.Context, string, string) error       { return nil }
func (mi mockPoliticianManager) PoliticianImages(context.Context, ...string) ([]string, error) {
	return nil, nil
}
func (mi mockPoliticianManager) All(context.Context) ([]Politician, string, error) {
	return []Politician{
		{Name: "Jack.Ma", Url: "xxx1", ID: "xxx11", Feature: []byte("xxxxx")},
		{Name: "Roby", Url: "xxx2", ID: "xxx22", Feature: []byte("xxxxx")},
		{Name: "Macxs", Url: "xxx3", ID: "xxx33", Feature: []byte("xxxxx")},
		{Name: "Msocre", Url: "xxx4", ID: "xxx44", Feature: []byte("xxxxx")},
		{Name: "Frank", Url: "xxx5", ID: "xxx55", Feature: []byte("xxxxx")},
	}, "", nil
}

func TestBjRPoliticianSearch(t *testing.T) {
	service, ctx := getMockContext(t)
	service.politician = &mockFaceSearch{
		Func: func(ctx context.Context, req interface{}, uid, utype uint32) (
			interface{}, error) {
			var ret evals.FaceSearchRespV2
			ret.Result.Confidences = append(
				ret.Result.Confidences, struct {
					Index  int     `json:"index"`
					Class  string  `json:"class"`
					Group  string  `json:"group"`
					Score  float32 `json:"score"`
					Sample struct {
						URL string   `json:"url"`
						Pts [][2]int `json:"pts"`
						ID  string   `json:"id"`
					} `json:"sample"`
				}{
					Class: "XXX",
					Score: 0.998,
				},
			)
			return ret, nil
		},
	}
	service._BjrunPoliticianManager = mockPoliticianManager{}
	service.iFacexSearch = mockDynamicFacexSearch{}
	service.iFaceDetect = mockFaceDetect{}
	service.iFaceFeatureV2 = mockFaceFeatureV2{}

	ctx.Exec(`
		post http://argus.ava.ai/v1/bjrun/politician/search
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
			"message": "",
			"result": {
				"detections": [
					{
						"value": {
							"name": $(name),
							"score":$(score),
							"review": $(review)
						}
					}
				]
			}
		}'
		equal $(code) 0
		equal $(name) XXX
		equal $(score) 0.998
		equal $(review) false
	`)
}

func TestBjRImageSearch(t *testing.T) {
	service, ctx := getMockContext(t)
	service._BjrunImageManager = mockImageManager{}
	service.iFeature = mockFeature{}
	service.iBjrunImageSearch = mockBjrunImageSearch{}
	service.iImageSearch = &mockImageSearch{
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
					Label: "ocean",
					Score: 0.998,
				},
			}
			return
		},
	}

	ctx.Exec(`
		post http://argus.ava.ai/v1/bjrun/image/search
		auth |authstub -uid 1 -utype 4|
		json '{
			"data": {
				"uri": "http://test.image.jpg"  
			}   
		}'
		ret 200
		echo $(resp.body)
		json '{
			"code": $(code),
			"result": [
				{
					"url":$(url),
					"label":$(label),
					"score":$(score)
				},
				{},
				{},
				{}
			]
		}'
		equal $(code) 0
		equal $(url) XXX
		equal $(label) ocean
		equal $(score) 0.998
	`)

}
