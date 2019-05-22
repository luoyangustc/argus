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
	"qiniu.com/argus/service/service/image"
	ipulp "qiniu.com/argus/service/service/image/pulp"
	pimagesync "qiniu.com/argus/service/service/image/pulp/image_sync"
	"qiniu.com/argus/service/service/video/vod/pulp"
	vod "qiniu.com/argus/service/service/video/vod/video"
)

const (
	VERSION = "1.0.0"
)

var (
	_DOC_PULP = scenario.OPDoc{
		Name:    "pulp",
		Version: VERSION,
		Desc:    []string{`色情识别`},
		Request: `
{
	...
	"ops": [
		{
			"op": "pulp",
			...
		},
		...
	]
}`,
		Response: `
{
	...
	"result": {
		"label": 2,
		"score": 0.9998751,
		"review": false
	}
	...
}`,
		RequestParam: []scenario.OpDocParam{
			{Name: "ops.[].op", Type: "string", Must: "Yes", Desc: `执行的推理cmd`},
		},
		ResponseParam: []scenario.OpDocParam{
			{Name: "label", Type: "int", Desc: `标签{0:色情，1:性感，2:正常}`},
			{Name: "score", Type: "float", Desc: `色情识别准确度`},
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

type Config ipulp.Config

func (c *Config) UnmarshalJSON(b []byte) error {
	type C Config
	c2 := C{}
	if err := json.Unmarshal(b, &c2); err != nil {
		return err
	}
	c1 := C(ipulp.DEFAULT)
	_ = mergo.Merge(&c2, &c1)
	*c = Config(c2)
	return nil
}

func Init(s scenario.VideoService, serviceID string) {
	var set = vod.GetSet(s, "qiniu.com/argus/service/service/video/vod/video")

	var config = Config(ipulp.DEFAULT)

	var (
		ss      pulp.PulpOP
		once    sync.Once
		pSet    biz.ServiceEvalSetter
		pfSet   biz.ServiceEvalSetter
		evalSet scenario.OPEvalSetter
	)

	newSS := func() {
		ss1, _ := pSet.Gen()
		ps := ss1.(ipulp.EvalPulpService)

		ss2, _ := pfSet.Gen()
		pfs := ss2.(ipulp.EvalPulpFilterService)

		s, _ := ipulp.NewPulpService(ipulp.Config(config), ps, pfs)
		ss = pulp.NewPulpOP(s)
	}

	opSet := set.RegisterOP(scenario.ServiceInfo{ID: serviceID, Version: VERSION},
		"pulp", &_DOC_PULP, func() interface{} {
			return pulp.NewOP(evalSet.Gen)
		})
	evalSet = opSet.RegisterEval(func() endpoint.Endpoint {
		once.Do(newSS)
		return ss.NewEval()
	})

	pSet = opSet.NewEval(
		biz.ServiceEvalInfo{Name: "evalPulp", Version: "1.1.0"},
		pimagesync.PulpEvalClient,
		func() middleware.ServiceEndpoints { return ipulp.EvalPulpEndpoints{} },
	).SetModel(ipulp.EVAL_PULP_CONFIG).GenId()

	pfSet = opSet.NewEval(
		biz.ServiceEvalInfo{Name: "evalPulpFilter", Version: "1.1.0"},
		pimagesync.PulpFilterEvalClient,
		func() middleware.ServiceEndpoints { return ipulp.EvalPulpFilterEndpoints{} },
	).SetModel(ipulp.EVAL_PULP_Filter_CONFIG).GenId()

	_ = opSet.GetConfig(context.Background(), &config)

}

var _ ipulp.PulpService = PulpService{}

type PulpService struct {
	Config
	ipulp.EvalPulpService
	ipulp.EvalPulpFilterService
}

func (s PulpService) Pulp(ctx context.Context, args ipulp.PulpReq) (res ipulp.PulpResp, err error) {
	// 视频和图片处理不同时需实现
	return
}

func (s PulpService) PulpCensor(context.Context, image.ImageCensorReq) (res image.SceneResult, err error) {
	// 视频和图片处理不同时需实现
	return
}
