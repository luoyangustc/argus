package video

import (
	"context"
	"encoding/json"
	"strings"

	"github.com/go-kit/kit/endpoint"

	"qiniu.com/argus/service/middleware"
	"qiniu.com/argus/service/scenario/biz"
	svideo "qiniu.com/argus/service/service/video"
)

type ServiceInfo struct {
	ID      string // pkg name
	Version string
}

type VideoService interface {
	Register(ServiceInfo, func(svideo.OPs) error, func()) ServiceSetter

	Prepare() error
	Init() error
	Info() error
	Run() error
}

type ServiceSetter interface {
	GetConfig(ctx context.Context, v interface{}) error
	Workspace() string
	Router() *Router

	GetOP(op string) OPSetter
	RegisterOP(ServiceInfo, string, *OPDoc, func() interface{}) OPSetter

	Init() error
	Run()
}

type OPSetter interface {
	GetConfig(ctx context.Context, v interface{}) error
	RegisterEval(func() endpoint.Endpoint) OPEvalSetter

	NewEvalDirect(biz.ServiceEvalInfo, *middleware.ServiceFactory) biz.ServiceEvalSetter
	NewEval(
		biz.ServiceEvalInfo,
		func(
			func(method string, respSample interface{}) (endpoint.Endpoint, error),
		) middleware.Service,
		func() middleware.ServiceEndpoints,
	) biz.ServiceEvalSetter
	Doc() []byte

	AddEvalsDeployModeOnGPU(
		device string,
		process [][]biz.ServiceEvalDeployProcess,
	) OPSetter
}

type OPEvalSetter interface {
	Gen() endpoint.Endpoint
}

func New() VideoService { return &videoServer{} }

type videoServer struct {
	config  Config
	router  *Router
	metrics *_MetricMiddleware
	evals   *biz.Evals

	ss []*serviceSetter
}

func (s *videoServer) Init() error {
	for _, set := range s.ss {
		if err := set.Init(); err != nil {
			return err
		}
	}
	return nil
}
func (s *videoServer) Register(info ServiceInfo, init func(svideo.OPs) error, run func()) ServiceSetter {
	conf, ok := s.config.Services[info.ID]
	if !ok {
		conf = &ServiceConfig{}
		if s.config.Services == nil {
			s.config.Services = make(map[string]*ServiceConfig)
		}
		s.config.Services[info.ID] = conf
	}
	setter := &serviceSetter{
		ServiceInfo:   info,
		ServiceConfig: conf,
		oss:           make(map[string]*opSetter),
		init:          init,
		run:           run,
		videoServer:   s,
	}
	s.ss = append(s.ss, setter)
	return setter
}

var _ ServiceSetter = &serviceSetter{}

type serviceSetter struct {
	ServiceInfo
	*ServiceConfig

	oss  map[string]*opSetter
	init func(svideo.OPs) error
	run  func()

	*videoServer
}

func (s *serviceSetter) GetConfig(ctx context.Context, v interface{}) error {
	if s.ServiceConfig.Service != nil {
		if err := json.Unmarshal(s.ServiceConfig.Service, v); err != nil {
			return err
		}
	}
	s.ServiceConfig.Service, _ = json.Marshal(v)
	return nil
}
func (s *serviceSetter) Workspace() string        { return s.config.Workspace }
func (s *serviceSetter) Router() *Router          { return s.router }
func (s *serviceSetter) GetOP(op string) OPSetter { return s.oss[op] }
func (s *serviceSetter) RegisterOP(
	info ServiceInfo, name string,
	doc *OPDoc,
	newF func() interface{}) OPSetter {
	conf, ok := s.ServiceConfig.OPs[info.ID]
	if !ok {
		conf = &OPConfig{}
		if s.ServiceConfig.OPs == nil {
			s.ServiceConfig.OPs = make(map[string]*OPConfig)
		}
		s.ServiceConfig.OPs[info.ID] = conf
	}
	setter := &opSetter{serviceSetter: s, OPConfig: conf, doc: doc, Name: name, NewF: newF}
	s.oss[name] = setter
	return setter
}
func (s *serviceSetter) Init() error {
	ops := svideo.OPs{}
	for name, os := range s.oss {
		ops[name] = os.NewF()
	}
	return s.init(ops)
}
func (s *serviceSetter) Run() {
	if s.run != nil {
		s.run()
	}
}

var _ OPSetter = &opSetter{}

type opSetter struct {
	*serviceSetter
	*OPConfig
	doc *OPDoc

	Name string
	NewF func() interface{}
}

func (s *opSetter) GetConfig(ctx context.Context, v interface{}) error {
	if s.OPConfig.OP != nil {
		if err := json.Unmarshal(s.OPConfig.OP, v); err != nil {
			return err
		}
	}
	s.OPConfig.OP, _ = json.Marshal(v)
	return nil
}
func (s *opSetter) RegisterEval(f func() endpoint.Endpoint) OPEvalSetter {
	return &opEvalSetter{NewEval: f,
		middleware: []endpoint.Middleware{
			s.metrics.Eval(s.Name),
		},
	}
}

func (s *opSetter) NewEvalDirect(biz.ServiceEvalInfo, *middleware.ServiceFactory) biz.ServiceEvalSetter {
	return nil
}

func (s *opSetter) NewEval(
	info biz.ServiceEvalInfo,
	f func(
		func(method string, respSample interface{}) (endpoint.Endpoint, error),
	) middleware.Service,
	newShell func() middleware.ServiceEndpoints,
) biz.ServiceEvalSetter {
	conf, ok := s.OPConfig.Evals[info.Name]
	if !ok {
		conf = &biz.ServiceEvalConfig{}
		if s.OPConfig.Evals == nil {
			s.OPConfig.Evals = make(map[string]*biz.ServiceEvalConfig)
		}
		s.OPConfig.Evals[info.Name] = conf
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
			AddLayer(s.metrics.NewEvalService(info.Name)),
	)
}

func (s *opSetter) Doc() []byte {
	if s.doc == nil {
		return []byte{}
	}
	bs, _ := s.doc.Marshal()
	return bs
}

func (s *opSetter) AddEvalsDeployModeOnGPU(
	device string,
	process [][]biz.ServiceEvalDeployProcess,
) OPSetter {
	s.OPConfig.EvalsDeploy.Modes = append(s.OPConfig.EvalsDeploy.Modes,
		biz.ServiceEvalDeployMode{
			Device:       device,
			ProcessOnGPU: process,
		},
	)
	return s
}

var _ OPEvalSetter = &opEvalSetter{}

type opEvalSetter struct {
	NewEval    func() endpoint.Endpoint
	middleware []endpoint.Middleware
}

func (s *opEvalSetter) Gen() endpoint.Endpoint {
	eval := s.NewEval()
	if len(s.middleware) == 0 {
		return eval
	}
	return endpoint.Chain(s.middleware[0], s.middleware[1:]...)(eval)
}
