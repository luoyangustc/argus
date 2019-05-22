package image_sync

import (
	"context"
	"testing"

	"github.com/go-kit/kit/endpoint"
	"github.com/stretchr/testify/assert"

	"qiniu.com/argus/service/middleware"
	"qiniu.com/argus/service/scenario/biz"
	pimage "qiniu.com/argus/service/service/image"
	"qiniu.com/argus/service/service/image/pulp" // 这个不应该import，如果需要重做一个mock service
	"qiniu.com/argus/service/transport"
	"qiniu.com/argus/utility/evals"
)

func TestMetrics(t *testing.T) {
	var conf = Config{Router: RouterConfig{Port: "6789"}}
	conf.Metrics.Namespace = "metrics_test"
	s := &imageServer{
		config:  conf,
		router:  newRouter(conf.Router),
		evals:   biz.NewEvals(),
		metrics: newMetricMiddleware(conf.Metrics),
	}

	type Config struct {
		pulp.Config
	}

	var config Config
	var es pulp.EvalPulpService
	var ess pulp.EvalPulpFilterService
	var eSet biz.ServiceEvalSetter
	var esSet biz.ServiceEvalSetter

	config.Config.PulpReviewThreshold = 0.89
	set := s.Register(
		ServiceInfo{Name: "pulp"},
		&middleware.ServiceFactory{
			New: func() middleware.Service {
				{
					ss, _ := eSet.Gen()
					es = ss.(pulp.EvalPulpService)

					ess1, _ := esSet.Gen()
					ess = ess1.(pulp.EvalPulpFilterService)
				}
				s, _ := pulp.NewPulpService(config.Config, es, ess)
				return s
			},
			NewShell: func() middleware.ServiceEndpoints { return pulp.PulpEndpoints{} },
		}).
		UpdateRouter(func(
			sf func() middleware.Service,
			imageParser func() pimage.IImageParse,
			path func(string) *ServiceRoute,
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
			path("/v1/pulp").Route().Methods("POST").Handler(transport.MakeHttpServer(
				func(ctx context.Context, req interface{}) (interface{}, error) { return endp().PulpEP(ctx, req) },
				evals.PulpReq{}))
			return nil
		})

	eSet = set.NewEval(
		biz.ServiceEvalInfo{Name: "evalPulp"},
		func(
			client func(method string, respSample interface{}) (endpoint.Endpoint, error),
		) middleware.Service {
			end, _ := client("POST", evals.PulpResp{})
			return pulp.EvalPulpEndpoints{EvalPulpEP: end}
		},
		func() middleware.ServiceEndpoints { return pulp.EvalPulpEndpoints{} },
	)

	esSet = set.NewEval(
		biz.ServiceEvalInfo{Name: "evalPulpFilter"},
		func(
			client func(method string, respSample interface{}) (endpoint.Endpoint, error),
		) middleware.Service {
			end, _ := client("POST", evals.PulpResp{})
			return pulp.EvalPulpFilterEndpoints{EvalPulpFilterEP: end}
		},
		func() middleware.ServiceEndpoints { return pulp.EvalPulpFilterEndpoints{} },
	)
	err := s.Init()
	assert.Nil(t, err)

}
