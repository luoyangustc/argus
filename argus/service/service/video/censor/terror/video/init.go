package video

import (
	"context"
	"encoding/json"
	"sync"

	"github.com/go-kit/kit/endpoint"
	"github.com/imdario/mergo"
	xlog "github.com/qiniu/xlog.v1"

	"qiniu.com/argus/service/middleware"
	"qiniu.com/argus/service/scenario/biz"
	scenario "qiniu.com/argus/service/scenario/video"
	. "qiniu.com/argus/service/service"
	iterror "qiniu.com/argus/service/service/image/terror"
	timagesync "qiniu.com/argus/service/service/image/terror/image_sync"
	"qiniu.com/argus/service/service/video/censor/terror"
	censor "qiniu.com/argus/service/service/video/censor/video"
	"qiniu.com/argus/utility/evals"
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

type Config iterror.Config

func (c *Config) UnmarshalJSON(b []byte) error {
	type C Config
	c2 := C{}
	if err := json.Unmarshal(b, &c2); err != nil {
		return err
	}
	c1 := C(iterror.DEFAULT)
	_ = mergo.Merge(&c2, &c1)
	*c = Config(c2)
	return nil
}

func Init(s scenario.VideoService, serviceID string) {
	var set = censor.GetSet(s, "qiniu.com/argus/service/service/video/censor/video")
	var config = Config(iterror.DEFAULT)

	var (
		sso1     terror.TerrorOP
		sso2     terror.TerrorOP
		once     sync.Once
		tmSet    biz.ServiceEvalSetter
		tdSet    biz.ServiceEvalSetter
		evalSet1 scenario.OPEvalSetter
		evalSet2 scenario.OPEvalSetter
	)

	newSS := func() {
		ss1, _ := tmSet.Gen()
		tms := ss1.(iterror.EvalTerrorMixupService)

		ss2, _ := tdSet.Gen()
		tds := ss2.(iterror.EvalTerrorDetectService)

		s1, _ := NewTerrorMainService(tms, tds)
		sso1 = terror.NewTerrorOP(s1)
		s2, _ := NewTerrorAuxiliaryService(tms)
		sso2 = terror.NewTerrorOP(s2)
	}

	opSet := set.RegisterOP(scenario.ServiceInfo{ID: serviceID, Version: VERSION},
		"terror", nil, func() interface{} {
			return terror.NewOP(evalSet1.Gen, evalSet2.Gen, iterror.Config(config).SugConfig)
		})
	evalSet1 = opSet.RegisterEval(func() endpoint.Endpoint {
		once.Do(newSS)
		return sso1.NewEval()
	})

	evalSet2 = opSet.RegisterEval(func() endpoint.Endpoint {
		once.Do(newSS)
		return sso2.NewEval()
	})

	tmSet = opSet.NewEval(
		biz.ServiceEvalInfo{Name: "evalTerrorMixup", Version: "1.1.0"},
		timagesync.TerrorMixupEvalClient,
		func() middleware.ServiceEndpoints { return iterror.EvalTerrorMixupEndpoints{} },
	).SetModel(iterror.EVAL_TERROR_MIXUP_CONFIG).GenId()

	tdSet = opSet.NewEval(
		biz.ServiceEvalInfo{Name: "evalTerrorDetect", Version: "1.1.0"},
		timagesync.TerrorDetEvalClient,
		func() middleware.ServiceEndpoints { return iterror.EvalTerrorDetectEndpoints{} },
	).SetModel(iterror.EVAL_TERROR_DET_CONFIG).GenId()

	_ = opSet.GetConfig(context.Background(), &config)

}

///////////////////////////////////////////////////////////////////////////

var _ terror.TerrorService = terrorMainService{}

type terrorMainService struct {
	iterror.EvalTerrorMixupService
	iterror.EvalTerrorDetectService
}

func NewTerrorMainService(
	s1 iterror.EvalTerrorMixupService,
	s2 iterror.EvalTerrorDetectService,
) (terror.TerrorService, error) {
	return terrorMainService{
		EvalTerrorMixupService:  s1,
		EvalTerrorDetectService: s2,
	}, nil
}

func (s terrorMainService) TerrorCensor(ctx context.Context, req iterror.SimpleReq) (res terror.TerrorResp, err error) {
	// 视频和图片处理不同时需实现---主帧

	var (
		xl    = xlog.FromContextSafe(ctx)
		mResp iterror.TerrorMixupResp
	)

	// 分类
	mResp, merr := s.EvalTerrorMixup(ctx, req)
	if merr != nil {
		err = ErrInternal(merr.Error())
		xl.Errorf("call terror_mixup failed. %v", err)
		return
	}

	if len(mResp.Result.Confidences) == 0 {
		xl.Errorf("unexpected terror_mixup resp:%v", mResp)
		err = ErrInternal(merr.Error())
		return
	}
	res.MixResp = &mResp
	if mResp.Result.Checkpoint == "endpoint" {
		return
	}
	// 继续检测
	var (
		dResp evals.TerrorDetectResp
		derr  error
	)
	dResp, derr = s.EvalTerrorDetect(ctx, req)
	if derr != nil {
		xl.Errorf("call /v1/eval/terror-detect error resp : %v", derr)
		err = ErrInternal(derr.Error())
		return
	}
	res.DetectResp = &dResp
	return
}

///////////////////////////////////////////////////////////////////////////
var _ terror.TerrorService = terrorAuxiliaryService{}

type terrorAuxiliaryService struct {
	iterror.EvalTerrorMixupService
}

func NewTerrorAuxiliaryService(
	s1 iterror.EvalTerrorMixupService,
) (terror.TerrorService, error) {
	return terrorAuxiliaryService{
		EvalTerrorMixupService: s1,
	}, nil
}

func (s terrorAuxiliaryService) TerrorCensor(ctx context.Context, req iterror.SimpleReq) (res terror.TerrorResp, err error) {
	// 视频和图片处理不同时需实现---辅助帧

	var (
		xl    = xlog.FromContextSafe(ctx)
		mResp iterror.TerrorMixupResp
	)

	// 分类
	mResp, merr := s.EvalTerrorMixup(ctx, req)
	if merr != nil {
		err = ErrInternal(merr.Error())
		xl.Errorf("call terror_mixup failed. %v", err)
		return
	}

	if len(mResp.Result.Confidences) == 0 {
		xl.Errorf("unexpected terror_mixup resp:%v", mResp)
		err = ErrInternal(merr.Error())
		return
	}
	res.MixResp = &mResp

	return
}
