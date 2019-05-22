package gate

import (
	"context"
	"strings"
	"sync"

	"github.com/qbox/ke-base/sdk/microservice"
	"github.com/qiniu/xlog.v1"
)

type Router interface {
	Match(context.Context, string) string
}

type RouterConfig struct {
	K8sConf struct {
		Username string `json:"user_name"`
		Password string `json:"password"`
		Host     string `json:"host"`
		Region   string `json:"region"`
	} `json:"k8s_configure"`
	AppsConf struct {
		Workspace string `json:"workspace"`
		Port      string `json:"port"`
		AppPrefix string `json:"app_prefix"`
	} `json:"apps_configure"`
}

type route struct {
	conf     RouterConfig
	client   microservice.Client
	services map[string]string
	*sync.Mutex
}

func NewRouter(cf RouterConfig, cl microservice.Client) Router {

	return &route{
		conf:     cf,
		client:   cl,
		services: make(map[string]string),
		Mutex:    &sync.Mutex{},
	}
}

func (r *route) update(ctx context.Context) {

	var (
		xl = xlog.FromContextSafe(ctx)
	)

	services, err := r.client.ListServiceV2(ctx, r.conf.AppsConf.Workspace)
	if err != nil {
		xl.Errorf("query k8s for app list error:%v", err)
		return
	}

	r.Lock()
	defer r.Unlock()
	for _, srv := range services {
		if _, ok := r.services[srv.Name]; ok {
			continue
		}
		if strings.HasPrefix(srv.Name, r.conf.AppsConf.AppPrefix) {
			//k8s remove the concept of app
			r.services[srv.Name] = srv.Name + ":" + r.conf.AppsConf.Port
		}
	}
	xl.Infof("services :%v", r.services)
}

func (r route) Match(ctx context.Context, app string) string {

	app = r.conf.AppsConf.AppPrefix + app //url 中不含ai4, 自动加
	if path, ok := r.services[app]; ok {
		return path
	}
	go r.update(ctx)
	return ""
}
