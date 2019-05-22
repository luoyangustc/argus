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
	"qiniu.com/argus/service/service/image/pulp"
	"qiniu.com/argus/service/transport"
	"qiniu.com/argus/utility/evals"
)

const (
	VERSION               = "1.1.0"
	EVAL_PULP_NAME        = "evalPulp"
	EVAL_PULP_FILTER_NAME = "evalPulpFilter"
)

var (
	_DOC_PULP = scenario.APIDoc{
		Version: VERSION,
		Desc:    []string{`用第三方服务和AtLab的剑皇服务做融合剑皇`, `每次输入一张图片，返回其内容是否含色情信息`},
		Request: `POST /v1/pulp  Http/1.1
Content-Type:application/json

{
	"data": 
		{
			"uri": "http://oayjpradp.bkt.clouddn.com/Audrey_Hepburn.jpg"
		},
	"params":
		{
			"limit":3
		}
}`,
		Response: ` 200 ok
Content-Type:application/json

{
	"code": ,
	"message": "",
	"result": {
		"label":1,
		"score":0.987,
		"review":false,
		"confidences":[
			{
				"index":1,
				"class":"sexy",
				"score":0.99999785
			},
			{
				"index":0,
				"class":"pulp",
				"score":4.0920222e-7
			},
			{
				"index":2,
				"class":"normal",
				"score":4.4148236e-9
			}
		]
	}
}`,
		RequestParam: []scenario.APIDocParam{
			{Name: "uri", Type: "string", Desc: "图片资源地址"},
			{Name: "params.limit", Type: "int", Desc: "是否显示详细信息；可选参数,1~3之间"},
		},
		ResponseParam: []scenario.APIDocParam{
			{Name: "code", Type: "int", Desc: "0:表示正确"},
			{Name: "message", Type: "string", Desc: "结果描述信息"},
			{Name: "result.label", Type: "int", Desc: "标签{0:色情，1:性感，2:正常}"},
			{Name: "result.score", Type: "float", Desc: "色情识别准确度"},
			{Name: "result.review", Type: "bool", Desc: "是否需要人工review"},
			{Name: "result.confidences", Type: "list", Desc: "图片打标信息列表,在params.limit>1时才返回"},
			{Name: "result.confidences.[].index", Type: "int", Desc: "类别编号, 即 0:pulp,1:sexy,2:norm"},
			{Name: "result.confidences.[].class", Type: "string", Desc: "图片内容鉴别结果，分为色情、性感或正常3类"},
			{Name: "result.confidences.[].score", Type: "float32", Desc: "将图片判别为某一类的准确度，取值范围0~1，1为准确度最高"},
		},
		ErrorMessage: []scenario.APIDocError{},
	}
)

var ON bool = false

func Import(serviceID string) func(interface{}) {
	ON = true
	return func(is interface{}) {
		Init(is.(scenario.ImageServer), serviceID)
	}
}

var (
	Set   scenario.ServiceSetter
	ESet  sbiz.ServiceEvalSetter
	ESSet sbiz.ServiceEvalSetter

	// 两卡下pulp的一种部署方式
	DeployMode2GPUCard = [][]sbiz.ServiceEvalDeployProcess{
		{
			{Name: EVAL_PULP_NAME, Num: 1},        // pulp GPU0
			{Name: EVAL_PULP_FILTER_NAME, Num: 1}, // pulp-filter GPU0
		},
		{
			// 其他服务 GPU1
		},
	}

	// 四卡下pulp的一种部署方式
	DeployMode4GPUCard = [][]sbiz.ServiceEvalDeployProcess{
		{
			{Name: EVAL_PULP_NAME, Num: 1},        // pulp GPU0
			{Name: EVAL_PULP_FILTER_NAME, Num: 1}, // pulp-filter GPU0
		},
		{
			// 其他服务 GPU1
		},
		{
			{Name: EVAL_PULP_NAME, Num: 1},        // pulp GPU2
			{Name: EVAL_PULP_FILTER_NAME, Num: 1}, // pulp-filter GPU2
		},
		{
			// 其他服务 GPU3
		},
	}
)

type Config pulp.Config

func (c *Config) UnmarshalJSON(b []byte) error {
	type C Config
	c2 := C{}
	if err := json.Unmarshal(b, &c2); err != nil {
		return err
	}
	c1 := C(pulp.DEFAULT)
	_ = mergo.Merge(&c2, &c1)
	*c = Config(c2)
	return nil
}

func Init(is scenario.ImageServer, serviceID string) {
	var config = Config(pulp.DEFAULT)

	var (
		es  pulp.EvalPulpService
		ess pulp.EvalPulpFilterService
	)
	// var eSet scenario.ServiceEvalSetter

	Set = is.Register(
		scenario.ServiceInfo{ID: serviceID, Name: "pulp", Version: VERSION},
		&middleware.ServiceFactory{
			New: func() middleware.Service {
				{
					es1, _ := ESet.Gen()
					es = es1.(pulp.EvalPulpService)

					ess1, _ := ESSet.Gen()
					ess = ess1.(pulp.EvalPulpFilterService)
				}
				s, _ := pulp.NewPulpService(pulp.Config(config), es, ess)
				return s
			},
			NewShell: func() middleware.ServiceEndpoints { return pulp.PulpEndpoints{} },
		}).
		UpdateRouter(func(
			sf func() middleware.Service,
			imageParser func() pimage.IImageParse,
			path func(string) *scenario.ServiceRoute,
		) error {
			endp := func() pulp.PulpEndpoints {
				svc := sf()
				endp, ok := svc.(pulp.PulpEndpoints)
				if !ok {
					svc, _ = middleware.MakeMiddleware(svc, pulp.PulpEndpoints{}, nil, nil)
					endp = svc.(pulp.PulpEndpoints)
				}
				return endp
			}
			path("/v1/pulp").Doc(_DOC_PULP).Route().
				Methods("POST").Handler(transport.MakeHttpServer(
				func(ctx context.Context, req0 interface{}) (interface{}, error) {
					req1, _ := req0.(evals.PulpReq)
					var req2 pulp.PulpReq
					var err error
					req2.Data.IMG, err = imageParser().ParseImage(ctx, req1.Data.URI)
					if err != nil {
						return nil, err
					}
					req2.Params.Limit = req1.Params.Limit
					return endp().PulpEP(ctx, req2)
				},
				evals.PulpReq{}))

			return nil
		})

	_ = Set.GetConfig(context.Background(), &config)

	ESet = Set.NewEval(
		sbiz.ServiceEvalInfo{Name: EVAL_PULP_NAME, Version: "1.1.0"},
		PulpEvalClient,
		func() middleware.ServiceEndpoints { return pulp.EvalPulpEndpoints{} },
	).SetModel(pulp.EVAL_PULP_CONFIG).GenId()

	ESSet = Set.NewEval(
		sbiz.ServiceEvalInfo{Name: EVAL_PULP_FILTER_NAME, Version: "1.1.0"},
		PulpFilterEvalClient,
		func() middleware.ServiceEndpoints { return pulp.EvalPulpFilterEndpoints{} },
	).SetModel(pulp.EVAL_PULP_Filter_CONFIG).GenId()

}

func PulpEvalClient(
	client func(method string, respSample interface{}) (endpoint.Endpoint, error),
) middleware.Service {
	end, _ := client("POST", evals.PulpResp{})
	return pulp.EvalPulpEndpoints{
		EvalPulpEP: func(ctx context.Context, req0 interface{}) (interface{}, error) {
			req1, _ := req0.(pulp.PulpReq)
			var req2 evals.PulpReq
			req2.Data.URI = string(req1.Data.IMG.URI)
			req2.Params.Limit = req1.Params.Limit
			return end(ctx, req2)
		}}
}

func PulpFilterEvalClient(
	client func(method string, respSample interface{}) (endpoint.Endpoint, error),
) middleware.Service {
	end, _ := client("POST", evals.PulpResp{})
	return pulp.EvalPulpFilterEndpoints{
		EvalPulpFilterEP: func(ctx context.Context, req0 interface{}) (interface{}, error) {
			req1, _ := req0.(pulp.PulpReq)
			var req2 evals.PulpReq
			req2.Data.URI = string(req1.Data.IMG.URI)
			req2.Params.Limit = req1.Params.Limit
			return end(ctx, req2)
		}}
}

func AddEvalsDeployMode() {
	if Set != nil {
		Set.AddEvalsDeployModeOnGPU("", DeployMode2GPUCard)
		Set.AddEvalsDeployModeOnGPU("", DeployMode4GPUCard)
	}
}
