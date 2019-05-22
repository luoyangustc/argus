package image_sync

import (
	"encoding/json"
	"io/ioutil"
	"net/http"
	"os"
	"path"
	"strings"

	"github.com/prometheus/client_golang/prometheus/promhttp"
	qlog "github.com/qiniu/log.v1"
	yaml "gopkg.in/yaml.v2"

	"qiniu.com/argus/service/middleware"
	"qiniu.com/argus/service/scenario/biz"
	pbiz "qiniu.com/argus/service/service/biz"
	pimage "qiniu.com/argus/service/service/image"
)

const (
	VERSION = "1.0.0"
)

func New() ImageServer { return &imageServer{} }

type ImageServer interface {
	Register(ServiceInfo, *middleware.ServiceFactory) ServiceSetter

	Prepare() error
	Init() error
	Info() error
	Run() error
}

var _ ImageServer = &imageServer{}

type imageServer struct {
	pimage.IImageParse
	ss   []*middleware.ServiceChain
	srvs []ServiceSetter

	config  Config
	evals   *biz.Evals
	metrics *_MetricMiddleware
	router  *Router
}

func (s *imageServer) Prepare() error {
	var config = LoadConfig()

	config.Version = VERSION

	var router = newRouter(config.Router)
	router.Handle("/metrics", promhttp.Handler())

	s.IImageParse = pimage.NewImageParser()
	s.config = config
	s.router = router
	s.evals = biz.NewEvals()
	s.metrics = newMetricMiddleware(config.Metrics)

	return nil
}

func (s *imageServer) Init() error {
	for _, ss := range s.ss {
		srv, err := ss.Gen()
		if err != nil {
			return err
		}
		_ = srv
	}
	for _, srv := range s.srvs {
		srv.Init()
	}
	return nil
}

func (s imageServer) Info() (err error) {
	wd, err := os.Getwd()
	if err != nil {
		return err
	}

	integrationPath := path.Join(wd, "integration")
	err = os.MkdirAll(integrationPath, 0755)
	if err != nil {
		return err
	}
	{
		// bs, err := json.Marshal(s.config)
		bs, err := json.MarshalIndent(s.config, "", "\t")
		if err != nil {
			return err
		}
		err = ioutil.WriteFile(path.Join(integrationPath, "config.json"), bs, 0644)
		if err != nil {
			return err
		}
	}

	// monitor part
	monitorPath := path.Join(integrationPath, "monitor")
	err = os.MkdirAll(monitorPath, 0755)
	if err != nil {
		return err
	}
	dashboardsPath := path.Join(monitorPath, "dashboards")
	err = os.MkdirAll(dashboardsPath, 0755)
	if err != nil {
		return err
	}
	for n, d := range s.metrics.MetricConfig.Dashborads {
		err = ioutil.WriteFile(path.Join(dashboardsPath, n)+".json", []byte(d), 0644)
		if err != nil {
			return err
		}
	}

	//model-configure
	modelsPath := path.Join(integrationPath, "modelconfig")
	err = os.MkdirAll(modelsPath, 0755)
	if err != nil {
		return err
	}
	m := map[string]pbiz.EvalModelConfig{}
	for _, s := range s.config.Services {
		for n1, e := range s.Evals {
			if strings.Contains(n1, "censor-") {
				continue
			}
			m[n1] = e.Model
		}
	}
	mcf, err := yaml.Marshal(m)
	if err != nil {
		return err
	}
	err = ioutil.WriteFile(path.Join(modelsPath, "service")+".yml", mcf, 0644)
	if err != nil {
		return err
	}

	//APIDoc
	bs, _ := s.router.Doc()
	_ = ioutil.WriteFile(path.Join(integrationPath, "api.md"), bs, 0644)

	qlog.Infof("dump info success")
	return nil
}

func (s imageServer) Run() error {
	return http.ListenAndServe("0.0.0.0:"+s.config.Router.Port, s.router.Router)
}
