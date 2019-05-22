package video

import (
	"context"
	"encoding/json"
	"sync"

	"github.com/go-kit/kit/endpoint"

	"github.com/imdario/mergo"

	"qiniu.com/argus/service/middleware"
	"qiniu.com/argus/service/scenario/biz"
	scenario "qiniu.com/argus/service/scenario/video"
	iterror "qiniu.com/argus/service/service/image/terror"
	timagesync "qiniu.com/argus/service/service/image/terror/image_sync"
	iterror_complex "qiniu.com/argus/service/service/image/terror_complex"
	"qiniu.com/argus/service/service/video/vod/terror_complex"
	vod "qiniu.com/argus/service/service/video/vod/video"
)

const (
	VERSION = "1.0.0"
)

var (
	_DOC_TERROR_COMPLEX = scenario.OPDoc{
		Name:    "terror_complex",
		Version: VERSION,
		Desc:    []string{`暴恐融合识别`},
		Request: `
{
	...
	"ops": [
		{
			"op": "terror_complex",
			"params": {
				"other": {
					"detail": <bool>
				}
				...
			}
			...
		},
		...
	]
}`,
		Response: `
{
	...
	"result": {
		"label": 1,
		"classes": [
			{
				"class": "xxx",
				"score": 0.94
			},
			{
				"class": "yyy",
				"score": 0.84
			}
		],
		"score": 0.9977869,
		"review": false
	}
	...
}`,
		RequestParam: []scenario.OpDocParam{
			{Name: "ops.[].op", Type: "string", Must: "Yes", Desc: `执行的推理cmd`},
			{Name: "ops.[].params.other.detail", Type: "bool", Must: "No", Desc: `是否显示详细信息，默认为false`},
		},
		ResponseParam: []scenario.OpDocParam{
			{Name: "label", Type: "int", Desc: `标签{0:正常，1:暴恐}`},
			{Name: "classes", Type: "list", Desc: `详细标签列表（指定detail=true的情况下返回）`},
			{Name: "classes.[].class", Type: "string", Desc: "标签类别"},
			{Name: "classes.[].score", Type: "float32", Desc: "标签类别准确度"},
			{Name: "score", Type: "float", Desc: `暴恐识别准确度`},
			{Name: "review", Type: "bool", Desc: `是否需要人工review`},
		},
	}
)

func Import(serviceID string) func(interface{}) {
	return func(s0 interface{}) {
		s := s0.(scenario.VideoService)
		Init(s, serviceID)
	}
}

type Config iterror_complex.Config

func (c *Config) UnmarshalJSON(b []byte) error {
	type C Config
	c2 := C{}
	if err := json.Unmarshal(b, &c2); err != nil {
		return err
	}
	c1 := C(iterror_complex.DEFAULT)
	_ = mergo.Merge(&c2, &c1)
	*c = Config(c2)
	return nil
}

func Init(s scenario.VideoService, serviceID string) {
	var set = vod.GetSet(s, "qiniu.com/argus/service/service/video/vod/video")
	var config = Config(iterror_complex.DEFAULT)

	var (
		ss      terror_complex.TerrorComplexOP
		once    sync.Once
		tmSet   biz.ServiceEvalSetter
		tdSet   biz.ServiceEvalSetter
		evalSet scenario.OPEvalSetter
	)

	newSS := func() {
		ss1, _ := tmSet.Gen()
		tms := ss1.(iterror.EvalTerrorMixupService)

		ss2, _ := tdSet.Gen()
		tds := ss2.(iterror.EvalTerrorDetectService)

		s, _ := iterror_complex.NewTerrorComplexService(iterror_complex.Config(config), tds, tms)
		ss = terror_complex.NewTerrorComplexOP(s)
	}

	opSet := set.RegisterOP(scenario.ServiceInfo{ID: serviceID, Version: VERSION},
		"terror_complex", &_DOC_TERROR_COMPLEX, func() interface{} {
			return terror_complex.NewOP(evalSet.Gen)
		})
	evalSet = opSet.RegisterEval(func() endpoint.Endpoint {
		once.Do(newSS)
		return ss.NewEval()
	})

	tmSet = opSet.NewEval(
		biz.ServiceEvalInfo{Name: "evalTerrorMixup", Version: "1.1.0"},
		timagesync.TerrorMixupEvalClient,
		func() middleware.ServiceEndpoints { return iterror.EvalTerrorMixupEndpoints{} },
	).SetModel(iterror_complex.EVAL_TERROR_MIXUP_CONFIG).GenId()

	tdSet = opSet.NewEval(
		biz.ServiceEvalInfo{Name: "evalTerrorDetect", Version: "1.1.0"},
		timagesync.TerrorDetEvalClient,
		func() middleware.ServiceEndpoints { return iterror.EvalTerrorDetectEndpoints{} },
	).SetModel(iterror_complex.EVAL_TERROR_DET_CONFIG).GenId()

	_ = opSet.GetConfig(context.Background(), &config)
}

type TerrorComplexService struct {
	Config
	iterror.EvalTerrorMixupService
	iterror.EvalTerrorDetectService
}

func (s TerrorComplexService) TerrorComplex(ctx context.Context, args iterror.TerrorReq) (ret iterror_complex.TerrorComplexResp, err error) {
	// 视频和图片处理不同时需实现
	return
}
