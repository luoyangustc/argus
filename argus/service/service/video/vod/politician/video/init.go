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
	ipolitician "qiniu.com/argus/service/service/image/politician"
	pimagesync "qiniu.com/argus/service/service/image/politician/image_sync"
	"qiniu.com/argus/service/service/video/vod/politician"
	vod "qiniu.com/argus/service/service/video/vod/video"
)

const (
	VERSION = "1.0.0"
)

var (
	_DOC_POLITICIAN = scenario.OPDoc{
		Name:    "politician",
		Version: VERSION,
		Desc:    []string{`政治人物搜索`},
		Request: `
{
	...
	"ops": [
		{
			"op": "politician",
			...
		},
		...
	]
}`,
		Response: `
{
	...
	"result": {
		"detections": [
			{
				"boundingBox":{
					"pts": [[1213,400],[205,400],[205,535],[1213,535]],
					"score":0.998
				},
				"value": {
					"name": "xx",
					"group": "xx",
					"score": 0.567,
					"review": false
				},
				"sample": {
					"url": "",
					"pts": [[1213,400],[205,400],[205,535],[1213,535]]
				}
			},
			...
		]
	}
	...
}`,
		RequestParam: []scenario.OpDocParam{
			{Name: "ops.[].op", Type: "string", Must: "Yes", Desc: `执行的推理cmd`},
		},
		ResponseParam: []scenario.OpDocParam{
			{Name: "detections", Type: "list", Desc: `检测到的人脸列表`},
			{Name: "detections.[].bounding_box.pts", Type: "list", Desc: `人脸所在图片中的位置，四点坐标值[左上，右上，右下，左下] 四点坐标框定的脸部`},
			{Name: "detections.[].bounding_box.score", Type: "float", Desc: `人脸的检测置信度人脸的检测置信度`},
			{Name: "detections.[].value.name", Type: "string", Desc: `相似政治人物姓名`},
			{Name: "detections.[].value.score", Type: "float", Desc: `人脸相似度`},
			{Name: "detections.[].value.group", Type: "string", Desc: `人物分组信息`},
			{Name: "detections.[].value.review", Type: "bool", Desc: `是否需要人工review`},
			{Name: "detections.[].sample.url", Type: "string", Desc: `该政治人物的示例图片信息`},
			{Name: "detections.[].sample.pts", Type: "list", Desc: `人脸在示例图片中的边框`},
		},
	}
)

func Import(serviceID string) func(interface{}) {
	return func(s0 interface{}) {
		s := s0.(scenario.VideoService)
		Init(s, serviceID)
	}
}

type Config ipolitician.Config

func (c *Config) UnmarshalJSON(b []byte) error {
	type C Config
	c2 := C{}
	if err := json.Unmarshal(b, &c2); err != nil {
		return err
	}
	c1 := C(ipolitician.DEFAULT)
	_ = mergo.Merge(&c2, &c1)
	*c = Config(c2)
	return nil
}

func Init(s scenario.VideoService, serviceID string) {
	var set = vod.GetSet(s, "qiniu.com/argus/service/service/video/vod/video")
	var config = Config(ipolitician.DEFAULT)

	var (
		ss      politician.PoliticianOP
		once    sync.Once
		fdSet   biz.ServiceEvalSetter
		ffSet   biz.ServiceEvalSetter
		pSet    biz.ServiceEvalSetter
		evalSet scenario.OPEvalSetter
	)

	newSS := func() {
		ss1, _ := fdSet.Gen()
		fds := ss1.(ipolitician.EvalFaceDetectService)

		ss2, _ := ffSet.Gen()
		ffs := ss2.(ipolitician.EvalFaceFeatureService)

		ss3, _ := pSet.Gen()
		ps := ss3.(ipolitician.EvalPoliticianService)

		s, _ := ipolitician.NewFaceSearchService(ipolitician.Config(config), fds, ffs, ps)
		ss = politician.NewPoliticianOP(s)
	}

	opSet := set.RegisterOP(scenario.ServiceInfo{ID: serviceID, Version: VERSION},
		"politician", &_DOC_POLITICIAN, func() interface{} {
			return politician.NewOP(evalSet.Gen)
		})
	evalSet = opSet.RegisterEval(func() endpoint.Endpoint {
		once.Do(newSS)
		return ss.NewEval()
	})

	fdSet = opSet.NewEval(
		biz.ServiceEvalInfo{Name: "evalFacexDetect", Version: "1.0.0"},
		pimagesync.FaceDetEvalClient,
		func() middleware.ServiceEndpoints { return ipolitician.EvalFaceDetectEndpoints{} },
	).SetModel(ipolitician.EVAL_FACE_DET_CONFIG).GenId()

	ffSet = opSet.NewEval(
		biz.ServiceEvalInfo{Name: "evalFacexFeatureV4", Version: "1.1.0"},
		pimagesync.FaceFeatureEvalClient,
		func() middleware.ServiceEndpoints { return ipolitician.EvalFaceFeatureEndpoints{} },
	).SetModel(ipolitician.EVAL_FACE_FEATURE_V4_CONFIG).GenId()

	pSet = opSet.NewEval(
		biz.ServiceEvalInfo{Name: "evalPolitician", Version: "1.1.0"},
		pimagesync.PoliticianEvalClient,
		func() middleware.ServiceEndpoints { return ipolitician.EvalPoliticianEndpoints{} },
	).SetModel(ipolitician.EVAL_POLITICIAN_CONFIG).GenId()

	_ = opSet.GetConfig(context.Background(), &config)

}

var _ ipolitician.FaceSearchService = PoliticianService{}

type PoliticianService struct {
	Config
	ipolitician.EvalFaceDetectService
	ipolitician.EvalFaceFeatureService
	ipolitician.EvalPoliticianService
}

func (s PoliticianService) FaceSearch(ctx context.Context, args ipolitician.Req) (ret ipolitician.FaceSearchResp, err error) {
	// 视频和图片处理不同时需实现
	return
}
func (s PoliticianService) PoliticianCensor(context.Context, image.ImageCensorReq) (res image.SceneResult, err error) {
	// 视频和图片处理不同时需实现
	return
}
