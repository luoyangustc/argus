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
	"qiniu.com/argus/service/service/video/censor/politician"
	censor "qiniu.com/argus/service/service/video/censor/video"
)

const (
	VERSION = "1.0.0"
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
	var set = censor.GetSet(s, "qiniu.com/argus/service/service/video/censor/video")
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
		"politician", nil, func() interface{} {
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
