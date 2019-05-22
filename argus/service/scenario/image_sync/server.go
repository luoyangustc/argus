package image_sync

import (
	"context"
	"encoding/json"
	"strings"

	"github.com/go-kit/kit/endpoint"

	"qiniu.com/argus/service/middleware"
	"qiniu.com/argus/service/scenario/biz"
	pimage "qiniu.com/argus/service/service/image"
)

type ServiceInfo struct {
	ID      string
	Name    string // pkg name
	Version string
}

type ServiceSetter interface {
	GetConfig(ctx context.Context, v interface{}) error
	NewEvalDirect(biz.ServiceEvalInfo, *middleware.ServiceFactory) biz.ServiceEvalSetter
	NewEval(
		biz.ServiceEvalInfo,
		func(
			func(method string, respSample interface{}) (endpoint.Endpoint, error),
		) middleware.Service,
		func() middleware.ServiceEndpoints,
	) biz.ServiceEvalSetter

	UpdateRouter(
		func(func() middleware.Service, func() pimage.IImageParse, func(string) *ServiceRoute) error,
	) ServiceSetter

	AddEvalsDeployModeOnGPU(
		device string,
		process [][]biz.ServiceEvalDeployProcess,
	) ServiceSetter // TODO 移至Evals部分，可以独立出来模型管理部分

	UpdateInit(func()) ServiceSetter

	Init()
}

////////////////////////////////////////////////////////////////////////////////

func (s *imageServer) Register(info ServiceInfo, sf *middleware.ServiceFactory) ServiceSetter {
	if len(info.ID) == 0 {
		info.ID = info.Name // 兼容逻辑
	}
	conf, ok := s.config.Services[info.ID]
	if !ok {
		conf = &ServiceConfig{}
		if s.config.Services == nil {
			s.config.Services = make(map[string]*ServiceConfig)
		}
		s.config.Services[info.ID] = conf
	}
	conf.Name = info.Name
	conf.Version = info.Version
	ss := &serviceSetter{
		ServiceInfo:   info,
		imageServer:   s,
		ServiceConfig: conf,
		routerLayer:   s.router.NewService(info.Name, s.metrics.NewImageParseService(info.Name, s.IImageParse)),
		merticsLayer:  s.metrics.NewService(info.Name),
	}
	s.ss = append(s.ss, middleware.NewServiceChain(sf).
		AddLayer(ss.merticsLayer).
		AddLayer(ss.routerLayer))
	s.srvs = append(s.srvs, ss)
	return ss
}

var _ ServiceSetter = &serviceSetter{}

type serviceSetter struct {
	*imageServer
	ServiceInfo
	*ServiceConfig

	routerLayer  *ServiceRouter
	merticsLayer serviceMetricMiddleware
	init         func()
}

func (s serviceSetter) GetConfig(ctx context.Context, v interface{}) error {
	if s.ServiceConfig.Service != nil {
		if err := json.Unmarshal(s.ServiceConfig.Service, v); err != nil {
			return err
		}
	}
	s.ServiceConfig.Service, _ = json.Marshal(v)
	return nil
}

func (s serviceSetter) NewEvalDirect(
	info biz.ServiceEvalInfo, sf *middleware.ServiceFactory,
) biz.ServiceEvalSetter {
	return biz.NewServiceEvalSetter(info, nil,
		middleware.
			NewServiceChain(sf).
			AddLayer(s.merticsLayer.newSubService(info.Name)),
	)
}

func (s serviceSetter) NewEval(
	info biz.ServiceEvalInfo,
	f func(
		client func(method string, respSample interface{}) (endpoint.Endpoint, error),
	) middleware.Service,
	newShell func() middleware.ServiceEndpoints,
) biz.ServiceEvalSetter {
	conf, ok := s.ServiceConfig.Evals[info.Name]
	if !ok {
		conf = &biz.ServiceEvalConfig{}
		if s.ServiceConfig.Evals == nil {
			s.ServiceConfig.Evals = make(map[string]*biz.ServiceEvalConfig)
		}
		s.ServiceConfig.Evals[info.Name] = conf
	}
	conf.Name = info.Name
	conf.Version = info.Version
	return biz.NewServiceEvalSetter(info, conf,
		middleware.
			NewServiceChain(
				&middleware.ServiceFactory{
					New: func() middleware.Service {
						return f(func(method string, respSample interface{},
						) (endpoint.Endpoint, error) {
							path := strings.TrimSuffix(conf.Host, "/") + "/" + strings.TrimPrefix(conf.Redirect, "/")
							return s.evals.Make(method, path, respSample)
						})
					},
					NewShell: newShell,
				},
			).
			AddLayer(s.merticsLayer.newSubService(info.Name)),
	)
}

func (s *serviceSetter) UpdateRouter(
	f func(func() middleware.Service, func() pimage.IImageParse, func(string) *ServiceRoute) error,
) ServiceSetter {
	s.routerLayer.Register(f)
	return s
}

func (s *serviceSetter) AddEvalsDeployModeOnGPU(
	device string,
	process [][]biz.ServiceEvalDeployProcess,
) ServiceSetter {
	s.ServiceConfig.EvalsDeploy.Modes = append(s.ServiceConfig.EvalsDeploy.Modes,
		biz.ServiceEvalDeployMode{
			Device:       device,
			ProcessOnGPU: process,
		},
	)
	return s
}

func (s *serviceSetter) UpdateInit(init func()) ServiceSetter {
	s.init = init
	return s
}

func (s *serviceSetter) Init() {
	if s.init != nil {
		s.init()
	}
}
