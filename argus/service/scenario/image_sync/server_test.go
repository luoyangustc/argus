package image_sync

import (
	"context"
	"encoding/json"
	"io/ioutil"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/go-kit/kit/endpoint"
	"github.com/stretchr/testify/assert"

	"qiniu.com/argus/service/middleware"
	"qiniu.com/argus/service/scenario/biz"
	pimage "qiniu.com/argus/service/service/image"
	"qiniu.com/argus/service/transport"
	"qiniu.com/argus/utility/evals"
)

type Foo interface {
	Foo(context.Context, evals.SimpleReq) (evals.SimpleReq, error)
}

var _ Foo = FooEndpoints{}

type FooEndpoints struct {
	FooEP endpoint.Endpoint
}

func (e FooEndpoints) Foo(_ context.Context, req evals.SimpleReq) (resp evals.SimpleReq, err error) {
	resp.Data.URI = "test_uri"
	return
}

var testConfig = Config{Router: RouterConfig{Port: "6789"}}

// var imgSvc = imageServer{
// 	config:  testConfig,
// 	router:  newRouter(testConfig.Router),
// 	evals:   newEvals(testConfig.Evlas),
// 	metrics: newMetricMiddleware(testConfig.Metrics),
// }

func TestImageServer(t *testing.T) {

	var imgSvc = imageServer{
		config:  testConfig,
		router:  newRouter(testConfig.Router),
		evals:   biz.NewEvals(),
		metrics: newMetricMiddleware(testConfig.Metrics),
	}

	imgSvc.Register(
		ServiceInfo{Name: "test"},
		&middleware.ServiceFactory{
			New: func() middleware.Service {
				return FooEndpoints{}
			},
			NewShell: func() middleware.ServiceEndpoints {
				return FooEndpoints{}
			},
		}).
		UpdateRouter(func(
			sf func() middleware.Service,
			_ func() pimage.IImageParse,
			path func(string) *ServiceRoute,
		) error {
			path("/v1/foo").Route().Methods("POST").Handler(transport.MakeHttpServer(
				func(ctx context.Context, req interface{}) (interface{}, error) {
					foo, _ := middleware.MakeMiddleware(sf(), FooEndpoints{}, nil, nil)
					endp := foo.(FooEndpoints)
					return endp.FooEP(ctx, req)
				}, evals.SimpleReq{}))
			return nil
		})

	_ = imgSvc.Init()

	var bd []byte
	var sp evals.SimpleReq
	srv := httptest.NewServer(imgSvc.router.Router)
	defer srv.Close()
	resp1, err := srv.Client().Post(srv.URL+"/v1/foo", "application/json", strings.NewReader(`{"data":{"uri":"test"}}`))
	defer func() {
		_ = resp1.Body.Close()
	}()

	assert.Nil(t, err)
	assert.Equal(t, imgSvc.config.Router.Port, "6789")
	assert.Equal(t, 1, len(imgSvc.ss))

	bd, err = ioutil.ReadAll(resp1.Body)
	_ = json.Unmarshal(bd, &sp)

	assert.Nil(t, err)
	assert.Equal(t, "test_uri", sp.Data.URI)
}
