package video

import (
	"context"
	"encoding/json"
	"sync"

	"github.com/go-kit/kit/endpoint"
	xlog "github.com/qiniu/xlog.v1"

	"github.com/imdario/mergo"

	"qiniu.com/argus/service/middleware"
	"qiniu.com/argus/service/scenario/biz"
	scenario "qiniu.com/argus/service/scenario/video"
	. "qiniu.com/argus/service/service"
	pimage "qiniu.com/argus/service/service/image"
	ipulp "qiniu.com/argus/service/service/image/pulp"
	pimagesync "qiniu.com/argus/service/service/image/pulp/image_sync"
	"qiniu.com/argus/service/service/video/censor/pulp"
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

type Config pimage.SugConfig

func (c *Config) UnmarshalJSON(b []byte) error {
	type C Config
	c2 := C{}
	if err := json.Unmarshal(b, &c2); err != nil {
		return err
	}
	c1 := C(ipulp.DEFAULT.SugConfig)
	_ = mergo.Merge(&c2, &c1)
	*c = Config(c2)
	return nil
}

func Init(s scenario.VideoService, serviceID string) {
	var set = censor.GetSet(s, "qiniu.com/argus/service/service/video/censor/video")

	var config = Config(ipulp.DEFAULT.SugConfig)

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

		s := NewPulpService(ps, pfs)
		ss = pulp.NewPulpOP(s)
	}

	opSet := set.RegisterOP(scenario.ServiceInfo{ID: serviceID, Version: VERSION},
		"pulp", nil, func() interface{} {
			return pulp.NewOP(evalSet.Gen, pimage.SugConfig(config))
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

var _ pulp.PulpService = new(pulpService)

type pulpService struct {
	ipulp.EvalPulpService
	ipulp.EvalPulpFilterService
}

func NewPulpService(
	ps ipulp.EvalPulpService,
	pfs ipulp.EvalPulpFilterService,
) *pulpService {
	return &pulpService{
		EvalPulpService:       ps,
		EvalPulpFilterService: pfs,
	}
}

func (s *pulpService) Pulp(ctx context.Context, req ipulp.PulpReq) (evals.PulpResp, error) {
	var (
		xl   = xlog.FromContextSafe(ctx)
		resp evals.PulpResp
	)

	resp, err := s.EvalPulpFilter(ctx, req)
	if err != nil {
		err = ErrInternal(err.Error())
		xl.Errorf("call pulp filter failed. %v", err)
		return resp, err
	}

	if resp.Result.Checkpoint == "endpoint" {
		return resp, err
	}

	resp, err = s.EvalPulp(ctx, req)
	if err != nil {
		err = ErrInternal(err.Error())
		xl.Errorf("call pulp failed. %v", err)
		return resp, err
	}

	return resp, err
}
