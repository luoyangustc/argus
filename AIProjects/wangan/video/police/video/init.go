package video

import (
	"context"
	"sync"

	"github.com/go-kit/kit/endpoint"
	ipolice "qiniu.com/argus/AIProjects/wangan/image/police"
	"qiniu.com/argus/AIProjects/wangan/image/police/image_sync"
	"qiniu.com/argus/AIProjects/wangan/video/police"
	"qiniu.com/argus/service/middleware"
	"qiniu.com/argus/service/scenario/biz"
	scenario "qiniu.com/argus/service/scenario/video"
	vod "qiniu.com/argus/service/service/video/vod/video"
)

const (
	VERSION = "1.0.0"
)

var (
	_DOC_POLICE = scenario.OPDoc{
		Name:    "police",
		Version: VERSION,
		Desc:    []string{`警察视频识别`},
		Request: `
{
	...
	"ops": [
		{
			"op": "police",
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
		"classes": ["xxx"],
		"score": 0.9977869,
	}
	...
}`,
		RequestParam: []scenario.OpDocParam{
			{Name: "ops.[].op", Type: "string", Must: "Yes", Desc: `执行的推理cmd`},
			{Name: "ops.[].params.other.detail", Type: "bool", Must: "No", Desc: `是否显示详细信息，默认为false`},
		},
		ResponseParam: []scenario.OpDocParam{
			{Name: "result.label", Type: "int", Desc: `标签{0:正常，1:涉警}`},
			{Name: "result.score", Type: "float", Desc: `涉警识别准确度`},
			{Name: "result.classes.[]", Type: "string", Desc: `详细类别`},
		},
	}
)

func Import(serviceID string) func(interface{}) {
	return func(s0 interface{}) {
		s := s0.(scenario.VideoService)
		Init(s, serviceID)
	}
}

type Config ipolice.Config

func Init(s scenario.VideoService, serviceID string) {
	var (
		ss      police.PoliceOP
		once    sync.Once
		set     = vod.GetSet(s, "qiniu.com/argus/service/service/video/vod/video")
		conf    Config
		pSet    biz.ServiceEvalSetter
		evalSet scenario.OPEvalSetter
	)

	newSS := func() {
		ss1, _ := pSet.Gen()
		srv, _ := ipolice.NewPoliceService(ipolice.Config(conf), ss1.(ipolice.EvalPoliceService))
		ss = police.NewPoliceOP(srv)
	}
	opSet := set.RegisterOP(scenario.ServiceInfo{ID: serviceID, Version: VERSION},
		"police", &_DOC_POLICE, func() interface{} {
			return police.NewOP(evalSet.Gen)
		})

	evalSet = opSet.RegisterEval(func() endpoint.Endpoint {
		once.Do(newSS)
		return ss.NewEval()
	})
	pSet = opSet.NewEval(
		biz.ServiceEvalInfo{Name: "evalPolice", Version: "1.0.0"},
		image_sync.PoliceEvalClient,
		func() middleware.ServiceEndpoints { return ipolice.EvalPoliceEndPoints{} },
	).SetModel(ipolice.EVAL_POLICE_CONFIG).GenId()

	_ = opSet.GetConfig(context.Background(), &conf)
}
