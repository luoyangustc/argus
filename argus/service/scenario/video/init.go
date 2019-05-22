package video

import (
	"context"
	"encoding/json"
	"io/ioutil"
	"net/http"
	"os"
	"path"

	"github.com/prometheus/client_golang/prometheus/promhttp"

	log "github.com/qiniu/log.v1"
	xlog "github.com/qiniu/xlog.v1"

	"qiniu.com/argus/service/scenario/biz"
)

func (s *videoServer) Prepare() error {

	var (
		config = LoadConfig()
		xl     = xlog.NewWith("main")
		ctx    = xlog.NewContext(context.Background(), xl)
	)
	_ = ctx
	log.SetOutputLevel(config.DebugLevel)

	s.metrics = newMetricMiddleware(config.Metrics)

	xl.Infof("%#v", config)

	s.ss = make([]*serviceSetter, 0)

	router := newRouter(config.Router)
	router.Handle("/metrics", promhttp.Handler())

	s.config = config
	s.router = router
	s.evals = biz.NewEvals()

	return nil
}

//----------------------------------------------------------------------------//

func (s videoServer) Info() (err error) {
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

	bs, _ := s.router.Doc()

	bs2 := []byte{}
	for _, ss := range s.ss {
		for _, set := range ss.oss {
			doc := set.Doc()
			bs2 = append(bs2, doc...)
		}
	}

	if len(bs2) > 0 {
		bs = append(bs, []byte(`
		# OPS
		`)...)
		bs = append(bs, bs2...)
	}

	if err = ioutil.WriteFile(path.Join(integrationPath, "api.md"), bs, 0644); err != nil {
		return err
	}

	log.Infof("dump info success")
	return nil
}
func (s videoServer) Run() error {
	for _, ss := range s.ss {
		ss.Run()
	}
	return http.ListenAndServe("0.0.0.0:"+s.config.Router.Port, s.router.Router)
}
