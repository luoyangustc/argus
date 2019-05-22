package politician

import (
	"context"
	"encoding/json"

	"github.com/go-kit/kit/endpoint"
	"github.com/imdario/mergo"

	"qiniu.com/argus/service/middleware"
	sbiz "qiniu.com/argus/service/scenario/biz"
	scenario "qiniu.com/argus/service/scenario/image_sync"
	pimage "qiniu.com/argus/service/service/image"
	"qiniu.com/argus/service/service/image/politician"
	"qiniu.com/argus/service/transport"
	"qiniu.com/argus/utility/evals"
)

const (
	VERSION                    = "1.1.0"
	EVAL_FACEX_DETECT_NAME     = "evalFacexDetect"
	EVAL_FACEX_FEAUTRE_V4_NAME = "evalFacexFeatureV4"
	EVAL_POLITICIAN_NAME       = "evalPolitician"
)

var (
	_DOC_POLITICIAN = scenario.APIDoc{
		Version: VERSION,
		Desc:    []string{`政治人物搜索，对输入图片识别检索是否存在政治人物`},
		Request: `POST /v1/face/search/politician Http/1.1
Content-Type: application/json

{
	"data": {
		"uri": "http://xx.com/xxx"
	}
}`,
		Response: ` 200 ok
Content-Type:application/json

{
	"code": 0,
	"message": "",
	"result": {
		"review": true,
		"detections": [
			{
				"boundingBox":{
					"pts": [[1213,400],[205,400],[205,535],[1213,535]],
					"score":0.998
				},
				"value": {
					"name": "xx",
					"group": "affairs_official_gov",
					"score":0.567,
					"review": true
				},
				"sample": {
					"url": ""http://xxx/xxx.jpg",
					"pts": [[1213,400],[205,400],[205,535],[1213,535]]
				}
			},
			{
				"boundingBox":{
					"pts": [[1109,500],[205,500],[205,535],[1109,535]],
					"score":0.98
				},
				"value": {
					"score":0.2207016,
					"review": false
				}
			}
		]
	}
}`,
		RequestParam: []scenario.APIDocParam{
			{Name: "uri", Type: "string", Desc: "图片资源地址"},
		},
		ResponseParam: []scenario.APIDocParam{
			{Name: "code", Type: "int", Desc: "0:表示正确"},
			{Name: "message", Type: "string", Desc: "结果描述信息"},
			{Name: "review", Type: "boolean", Desc: "True或False,图片是否需要人工review, 只要有一个value.review为True,则此字段为True"},
			{Name: "boundingBox", Type: "map", Desc: "人脸边框信息"},
			{Name: "boundingBox.pst", Type: "list", Desc: "人脸边框在图片中的位置[左上，右上，右下，左下]"},
			{Name: "boundingBox.score", Type: "float", Desc: "人脸位置检测准确度"},
			{Name: "value.name", Type: "string", Desc: "检索得到的政治人物姓名,value.score < 0.4时未找到相似人物,没有这个字段"},
			{Name: "value.group", Type: "string", Desc: "人物分组信息，总共有8个组，value.score < 0.4时未找到相似人物,没有这个字段"},
			{Name: "value.review", Type: "boolean", Desc: "True或False,当前人脸识别结果是否需要人工review"},
			{Name: "value.score", Type: "float", Desc: "0~1,检索结果的可信度, 0.38 < value.score < 0.42 时 value.review 为true"},
			{Name: "sample", Type: "object", Desc: "该政治人物的示例图片信息，value.score < 0.4时未找到相似人物, 没有这个字段"},
			{Name: "sample.url", Type: "string", Desc: "该政治人物的示例图片"},
			{Name: "sample.pts", Type: "list", Desc: "人脸在示例图片中的边框"},
		},
		ErrorMessage: []scenario.APIDocError{},
		Appendix: []string{`分组标签取值及说明：

| 标签 | 说明 |
| :--- | :--- |
| domestic_statesman | 国内政治人物 |
| foreign_statesman | 国外政治人物 |
| affairs_official_gov | 落马官员（政府) |
| affairs_official_ent | 落马官员（企事业）|
| anti_china_people | 反华分子 |
| terrorist | 恐怖分子 |
| affairs_celebrity | 劣迹艺人 |
| chinese_martyr | 烈士 |
`},
	}
)

var ON bool = false

func Import(serviceID string) func(interface{}) {
	ON = true
	return func(is interface{}) {
		Init(is.(scenario.ImageServer), serviceID)
	}
}

type Config politician.Config

func (c *Config) UnmarshalJSON(b []byte) error {
	type C Config
	c2 := C{}
	if err := json.Unmarshal(b, &c2); err != nil {
		return err
	}
	c1 := C(politician.DEFAULT)
	_ = mergo.Merge(&c2, &c1)
	*c = Config(c2)
	return nil
}

var (
	Set   scenario.ServiceSetter
	ESet1 sbiz.ServiceEvalSetter
	ESet2 sbiz.ServiceEvalSetter
	ESet3 sbiz.ServiceEvalSetter

	// 两卡下politician的一种部署方式
	DeployMode2GPUCard = [][]sbiz.ServiceEvalDeployProcess{
		{
			{Name: EVAL_FACEX_DETECT_NAME, Num: 1},     // facex-detect GPU0
			{Name: EVAL_FACEX_FEAUTRE_V4_NAME, Num: 1}, // facex-feature-v4 GPU0
		},
		{
			// 其他服务 GPU1
		},
	}

	// 四卡下politician的一种部署方式
	DeployMode4GPUCard = [][]sbiz.ServiceEvalDeployProcess{
		{
			{Name: EVAL_FACEX_DETECT_NAME, Num: 1},     // facex-detect GPU0
			{Name: EVAL_FACEX_FEAUTRE_V4_NAME, Num: 1}, // facex-feature-v4 GPU0
		},
		{
			// 其他服务 GPU1
		},
		{
			{Name: EVAL_FACEX_DETECT_NAME, Num: 1},     // facex-detect GPU2
			{Name: EVAL_FACEX_FEAUTRE_V4_NAME, Num: 1}, // facex-feature-v4 GPU2
		},
		{
			// 其他服务 GPU3
		},
	}
)

func Init(is scenario.ImageServer, serviceID string) {
	var config = Config(politician.DEFAULT)

	var ps1 politician.EvalFaceDetectService
	var ps2 politician.EvalFaceFeatureService
	var ps3 politician.EvalPoliticianService

	Set = is.Register(
		scenario.ServiceInfo{ID: serviceID, Name: "politician", Version: VERSION},
		&middleware.ServiceFactory{
			New: func() middleware.Service {
				{
					ss1, _ := ESet1.Gen()
					ps1 = ss1.(politician.EvalFaceDetectService)

					ss2, _ := ESet2.Gen()
					ps2 = ss2.(politician.EvalFaceFeatureService)

					ss3, _ := ESet3.Gen()
					ps3 = ss3.(politician.EvalPoliticianService)
				}
				s, _ := politician.NewFaceSearchService(politician.Config(config), ps1, ps2, ps3)
				return s
			},
			NewShell: func() middleware.ServiceEndpoints { return politician.FaceSearchEndpoints{} },
		}).
		UpdateRouter(func(
			sf func() middleware.Service,
			imageParser func() pimage.IImageParse,
			path func(string) *scenario.ServiceRoute,
		) error {
			endp := func() politician.FaceSearchEndpoints {
				svc := sf()
				endp, ok := svc.(politician.FaceSearchEndpoints)
				if !ok {
					svc, _ = middleware.MakeMiddleware(svc, politician.FaceSearchEndpoints{}, nil, nil)
					endp = svc.(politician.FaceSearchEndpoints)
				}
				return endp
			}
			path("/v1/face/search/politician").Doc(_DOC_POLITICIAN).Route().
				Methods("POST").Handler(transport.MakeHttpServer(
				func(ctx context.Context, req0 interface{}) (interface{}, error) {
					req1, _ := req0.(evals.SimpleReq)
					var req2 politician.Req
					var err error
					req2.Data.IMG, err = imageParser().ParseImage(ctx, req1.Data.URI)
					if err != nil {
						return nil, err
					}
					return endp().FaceSearchEP(ctx, req2)
				},
				evals.SimpleReq{}))

			return nil
		})

	_ = Set.GetConfig(context.Background(), &config)

	ESet1 = Set.NewEval(
		sbiz.ServiceEvalInfo{Name: EVAL_FACEX_DETECT_NAME, Version: "1.0.0"},
		FaceDetEvalClient,
		func() middleware.ServiceEndpoints { return politician.EvalFaceDetectEndpoints{} },
	).SetModel(politician.EVAL_FACE_DET_CONFIG).GenId()

	ESet2 = Set.NewEval(
		sbiz.ServiceEvalInfo{Name: EVAL_FACEX_FEAUTRE_V4_NAME, Version: "1.1.0"},
		FaceFeatureEvalClient,
		func() middleware.ServiceEndpoints { return politician.EvalFaceFeatureEndpoints{} },
	).SetModel(politician.EVAL_FACE_FEATURE_V4_CONFIG).GenId()

	ESet3 = Set.NewEval(
		sbiz.ServiceEvalInfo{Name: EVAL_POLITICIAN_NAME, Version: "1.1.0"},
		PoliticianEvalClient,
		func() middleware.ServiceEndpoints { return politician.EvalPoliticianEndpoints{} },
	).SetModel(politician.EVAL_POLITICIAN_CONFIG).GenId()
}

func FaceDetEvalClient(
	client func(method string, respSample interface{}) (endpoint.Endpoint, error),
) middleware.Service {
	end, _ := client("POST", evals.FaceDetectResp{})
	return politician.EvalFaceDetectEndpoints{
		EvalFaceDetectEP: func(ctx context.Context, req0 interface{}) (interface{}, error) {
			req1, _ := req0.(politician.Req)
			var req2 evals.SimpleReq
			req2.Data.URI = string(req1.Data.IMG.URI)
			return end(ctx, req2)
		}}
}

func FaceFeatureEvalClient(
	client func(method string, respSample interface{}) (endpoint.Endpoint, error),
) middleware.Service {
	end, _ := client("POST", []byte{})
	return politician.EvalFaceFeatureEndpoints{
		EvalFaceFeatureEP: func(ctx context.Context, req0 interface{}) (interface{}, error) {
			req1, _ := req0.(politician.FaceReq)
			var req2 evals.FaceReq
			req2.Data.URI = string(req1.Data.IMG.URI)
			req2.Data.Attribute = req1.Data.Attribute
			return end(ctx, req2)
		}}
}

func PoliticianEvalClient(
	client func(method string, respSample interface{}) (endpoint.Endpoint, error),
) middleware.Service {
	end, _ := client("POST", evals.FaceSearchRespV2{})
	return politician.EvalPoliticianEndpoints{EvalPoliticianEP: end}
}

func AddEvalsDeployMode() {
	if Set != nil {
		Set.AddEvalsDeployModeOnGPU("", DeployMode2GPUCard)
		Set.AddEvalsDeployModeOnGPU("", DeployMode4GPUCard)
	}
}
