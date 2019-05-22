package image_sync

import (
	"context"
	"encoding/json"

	"github.com/go-kit/kit/endpoint"
	"github.com/imdario/mergo"

	"qiniu.com/argus/service/middleware"
	sbiz "qiniu.com/argus/service/scenario/biz"
	scenario "qiniu.com/argus/service/scenario/image_sync"
	pimage "qiniu.com/argus/service/service/image"
	"qiniu.com/argus/service/service/image/terror"
	"qiniu.com/argus/service/transport"
	"qiniu.com/argus/utility/evals"
)

const (
	VERSION                 = "1.1.0"
	EVAL_TERROR_MIXUP_NAME  = "evalTerrorMixup"
	EVAL_TERROR_DETECT_NAME = "evalTerrorDetect"
)

var (
	_DOC_TERROR = scenario.APIDoc{
		Version: VERSION,
		Desc:    []string{`用检测暴恐识别和分类暴恐识别方法做融合暴恐识别`, `每次输入一张图片，返回其内容是否含暴恐信息`},
		Request: `POST /v1/terror  Http/1.1
Content-Type: application/json
{
	"data": {
		"uri": "http://xxx/xxx.jpg"
	},
	"params": {
		"detail": true
	}
}`,
		Response: ` 200 ok
Content-Type:application/json
{
	"code": 0,
	"message": "",
	"result": {
		"label":0,
		"class":"normal",
		"score":0.9996768,
		"review":false
	}
}`,
		RequestParam: []scenario.APIDocParam{
			{Name: "uri", Type: "string", Desc: "图片资源地址"},
			{Name: "params.detail", Type: "bool", Desc: "是否显示详细信息；可选参数"},
		},
		ResponseParam: []scenario.APIDocParam{
			{Name: "code", Type: "int", Desc: "0:表示正确"},
			{Name: "message", Type: "string", Desc: "结果描述信息"},
			{Name: "result.label", Type: "int", Desc: "标签{0:正常，1:暴恐}"},
			{Name: "result.class", Type: "string", Desc: "标签（指定detail=true的情况下返回）"},
			{Name: "result.score", Type: "float", Desc: "暴恐识别准确度"},
			{Name: "result.review", Type: "bool", Desc: "是否需要人工review"},
		},
		ErrorMessage: []scenario.APIDocError{},
		Appendix: []string{`暴恐标签取值及说明：

| 标签 | 说明 |
| :--- | :--- |
| illegal_flag | 违规旗帜 |
| knives | 刀 |
| guns | 枪 |
| anime_knives | 二次元刀 |
| anime_guns | 二次元枪 |
| bloodiness | 血腥 |
| self_burning | 自焚 |
| beheaded | 行刑斩首 |
| march_crowed | 非法集会 |
| fight_police | 警民冲突 |
| fight_person | 打架斗殴 |
| special_characters | 特殊字符 |
| anime_bloodiness | 二次元血腥 |
| special_clothing | 特殊着装 |
| normal | 正常 |
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

type Config terror.Config

func (c *Config) UnmarshalJSON(b []byte) error {
	type C Config
	c2 := C{}
	if err := json.Unmarshal(b, &c2); err != nil {
		return err
	}
	c1 := C(terror.DEFAULT)
	_ = mergo.Merge(&c2, &c1)
	*c = Config(c2)
	return nil
}

var (
	Set   scenario.ServiceSetter
	ESet1 sbiz.ServiceEvalSetter
	ESet2 sbiz.ServiceEvalSetter

	// 两卡下terror的一种部署方式
	DeployMode2GPUCard = [][]sbiz.ServiceEvalDeployProcess{
		{
			{Name: EVAL_TERROR_MIXUP_NAME, Num: 1},  // terror-mixup GPU0
			{Name: EVAL_TERROR_DETECT_NAME, Num: 1}, // terror-detect GPU0
		},
		{
			// 其他服务 GPU1
		},
	}

	// 四卡下terror的一种部署方式
	DeployMode4GPUCard = [][]sbiz.ServiceEvalDeployProcess{
		{
			{Name: EVAL_TERROR_MIXUP_NAME, Num: 1},  // terror-mixup GPU0
			{Name: EVAL_TERROR_DETECT_NAME, Num: 1}, // terror-detect GPU0
		},
		{
			// 其他服务 GPU1
		},
		{
			{Name: EVAL_TERROR_MIXUP_NAME, Num: 1},  // terror-mixup GPU2
			{Name: EVAL_TERROR_DETECT_NAME, Num: 1}, // terror-detect GPU2
		},
		{
			// 其他服务 GPU3
		},
	}
)

func Init(is scenario.ImageServer, serviceID string) {
	var config = Config(terror.DEFAULT)

	var ts1 terror.EvalTerrorMixupService
	var ts2 terror.EvalTerrorDetectService

	Set = is.Register(
		scenario.ServiceInfo{ID: serviceID, Name: "terror", Version: VERSION},
		&middleware.ServiceFactory{
			New: func() middleware.Service {
				{
					ss1, _ := ESet1.Gen()
					ts1 = ss1.(terror.EvalTerrorMixupService)

					ss2, _ := ESet2.Gen()
					ts2 = ss2.(terror.EvalTerrorDetectService)

				}
				s, _ := terror.NewTerrorService(terror.Config(config), ts1, ts2)
				return s
			},
			NewShell: func() middleware.ServiceEndpoints { return terror.TerrorEndpoints{} },
		}).
		UpdateRouter(func(
			sf func() middleware.Service,
			imageParser func() pimage.IImageParse,
			path func(string) *scenario.ServiceRoute,
		) error {
			endp := func() terror.TerrorEndpoints {
				svc := sf()
				endp, ok := svc.(terror.TerrorEndpoints)
				if !ok {
					svc, _ = middleware.MakeMiddleware(svc, terror.TerrorEndpoints{}, nil, nil)
					endp = svc.(terror.TerrorEndpoints)
				}
				return endp
			}

			type Req struct {
				Data struct {
					URI string `json:"uri"`
				} `json:"data"`
				Params struct {
					Detail bool `json:"detail"`
				} `json:"params"`
			}

			path("/v1/terror").Doc(_DOC_TERROR).Route().
				Methods("POST").Handler(transport.MakeHttpServer(
				func(ctx context.Context, req0 interface{}) (interface{}, error) {
					req1, _ := req0.(Req)
					var req2 terror.TerrorReq
					req2.Params.Detail = req1.Params.Detail
					var err error
					req2.Data.IMG, err = imageParser().ParseImage(ctx, req1.Data.URI)
					if err != nil {
						return nil, err
					}
					return endp().TerrorEP(ctx, req2)
				},
				Req{}))

			return nil
		})

	_ = Set.GetConfig(context.Background(), &config)

	ESet1 = Set.NewEval(
		sbiz.ServiceEvalInfo{Name: EVAL_TERROR_MIXUP_NAME, Version: "1.1.0"},
		TerrorMixupEvalClient,
		func() middleware.ServiceEndpoints { return terror.EvalTerrorMixupEndpoints{} },
	).SetModel(terror.EVAL_TERROR_MIXUP_CONFIG).GenId()

	ESet2 = Set.NewEval(
		sbiz.ServiceEvalInfo{Name: EVAL_TERROR_DETECT_NAME, Version: "1.1.0"},
		TerrorDetEvalClient,
		func() middleware.ServiceEndpoints { return terror.EvalTerrorDetectEndpoints{} },
	).SetModel(terror.EVAL_TERROR_DET_CONFIG).GenId()

}

func TerrorDetEvalClient(
	client func(method string, respSample interface{}) (endpoint.Endpoint, error),
) middleware.Service {
	end, _ := client("POST", evals.TerrorDetectResp{})
	return terror.EvalTerrorDetectEndpoints{
		EvalTerrorDetectEP: func(ctx context.Context, req0 interface{}) (interface{}, error) {
			req1, _ := req0.(terror.SimpleReq)
			var req2 evals.SimpleReq
			req2.Data.URI = string(req1.Data.IMG.URI)
			return end(ctx, req2)
		}}
}

func TerrorMixupEvalClient(
	client func(method string, respSample interface{}) (endpoint.Endpoint, error),
) middleware.Service {
	end, _ := client("POST", terror.TerrorMixupResp{})
	return terror.EvalTerrorMixupEndpoints{
		EvalTerrorMixupEP: func(ctx context.Context, req0 interface{}) (interface{}, error) {
			req1, _ := req0.(terror.SimpleReq)
			var req2 evals.SimpleReq
			req2.Data.URI = string(req1.Data.IMG.URI)
			return end(ctx, req2)
		}}
}

func AddEvalsDeployMode() {
	if Set != nil {
		Set.AddEvalsDeployModeOnGPU("", DeployMode2GPUCard)
		Set.AddEvalsDeployModeOnGPU("", DeployMode4GPUCard)
	}
}
