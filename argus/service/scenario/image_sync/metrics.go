package image_sync

import (
	"context"
	"strconv"
	"time"

	"github.com/go-kit/kit/endpoint"
	"github.com/go-kit/kit/metrics"
	kitprometheus "github.com/go-kit/kit/metrics/prometheus"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/qiniu/http/httputil.v1"

	"qiniu.com/argus/service/middleware"
	"qiniu.com/argus/service/scenario/image_sync/dashboard"
	. "qiniu.com/argus/service/service"
	pimage "qiniu.com/argus/service/service/image"
)

type MetricConfig struct {
	Namespace string `json:"namespace"`
	Subsystem string `json:"subsystem"`

	// Metric List 需要暴露?
	Dashborads map[string]dashboard.Dashboard `json:"dashboards"` // 场景下的监控 dashboard list
}

type _MetricMiddleware struct {
	MetricConfig

	requestCount           metrics.Counter
	responseHistogram      metrics.Histogram
	subRequestCount        metrics.Counter
	subResponseHistogram   metrics.Histogram
	imageRequestCount      metrics.Counter
	imageResponseHistogram metrics.Histogram
	// logger log.Logger
}

func newMetricMiddleware(conf MetricConfig) *_MetricMiddleware {
	if len(conf.Namespace) == 0 {
		conf.Namespace = "qiniu_ai"
	}
	if len(conf.Subsystem) == 0 {
		conf.Subsystem = "image"
	}
	conf.Dashborads = dashboard.Load()

	return &_MetricMiddleware{
		MetricConfig: conf,

		requestCount: kitprometheus.NewCounterFrom(
			prometheus.CounterOpts{
				Namespace: conf.Namespace,
				Subsystem: conf.Subsystem,
				Name:      "argus_service_request_count",
				Help:      "Number of requests received.",
			},
			[]string{"service", "api"},
		),

		responseHistogram: kitprometheus.NewHistogramFrom(
			prometheus.HistogramOpts{
				Namespace: conf.Namespace,
				Subsystem: conf.Subsystem,
				Name:      "argus_service_response_time",
				Help:      "response time of requests",
				Buckets: []float64{
					0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
					1, 2, 3, 4, 5, 6, 7, 8, 9,
					10, 20, 30, 40, 50, 60,
				}, // time.second
			},
			[]string{"service", "api", "code"},
		),

		subRequestCount: kitprometheus.NewCounterFrom(
			prometheus.CounterOpts{
				Namespace: conf.Namespace,
				Subsystem: conf.Subsystem,
				Name:      "argus_sub_service_request_count",
				Help:      "Number of sub requests received.",
			},
			[]string{"sub_service", "service"},
		),

		subResponseHistogram: kitprometheus.NewHistogramFrom(
			prometheus.HistogramOpts{
				Namespace: conf.Namespace,
				Subsystem: conf.Subsystem,
				Name:      "argus_sub_service_response_time",
				Help:      "Number of response responsed.",
			},
			[]string{"sub_service", "service", "code"},
		),

		imageRequestCount: kitprometheus.NewCounterFrom(
			prometheus.CounterOpts{
				Namespace: conf.Namespace,
				Subsystem: conf.Subsystem,
				Name:      "image_parse_request_count",
				Help:      "Number of requests received.",
			},
			[]string{"service"},
		),

		imageResponseHistogram: kitprometheus.NewHistogramFrom(
			prometheus.HistogramOpts{
				Namespace: conf.Namespace,
				Subsystem: conf.Subsystem,
				Name:      "image_parse_response_time",
				Help:      "response time of requests",
				Buckets: []float64{
					0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
					1, 2, 3, 4, 5, 6, 7, 8, 9,
					10, 20, 30, 40, 50, 60,
				}, // time.second
			},
			[]string{"service", "code"},
		),
	}
}

func detectCode(err error) int {
	if err == nil {
		return 200
	}
	info, ok := err.(DetectErrorer)
	if ok {
		httpCode, _, _ := info.DetectError()
		return httpCode
	} else {
		return httputil.DetectCode(err)
	}
}

type imageParseServiceFunc func(context.Context, string) (pimage.Image, error)

func (s imageParseServiceFunc) ParseImage(ctx context.Context, uri string) (pimage.Image, error) {
	return s(ctx, uri)
}

func (m _MetricMiddleware) NewImageParseService(serviceName string, parser pimage.IImageParse) pimage.IImageParse {
	return imageParseServiceFunc(func(ctx context.Context, uri string) (pimage.Image, error) {
		var code string
		defer func(begin time.Time) {
			m.imageRequestCount.
				With(
					"service", serviceName,
				).Add(1)
			m.imageResponseHistogram.
				With(
					"service", serviceName,
					"code", code,
				).Observe(time.Since(begin).Seconds())
		}(time.Now())
		resp, err := parser.ParseImage(ctx, uri)
		codeInt := detectCode(err)
		code = strconv.Itoa(codeInt)
		return resp, err
	})
}

func (m *_MetricMiddleware) NewService(name string) serviceMetricMiddleware {
	return serviceMetricMiddleware{
		_MetricMiddleware: m,
		serviceName:       name,
	}
}

type serviceMetricMiddleware struct {
	*_MetricMiddleware
	serviceName string
}

func (m serviceMetricMiddleware) New(svc middleware.Service, endpoints middleware.ServiceEndpoints) (middleware.Service, error) {
	return middleware.MakeMiddlewareFactory(nil,
		func(methodName string, service middleware.Service, defaultEndpoint func() endpoint.Endpoint) endpoint.Endpoint {
			e := defaultEndpoint()
			return func(ctx context.Context, request interface{}) (response interface{}, err error) {
				var code string
				defer func(begin time.Time) {
					m.requestCount.
						With(
							"service", m.serviceName,
							"api", methodName,
						).Add(1)
					m.responseHistogram.
						With(
							"service", m.serviceName,
							"api", methodName,
							"code", code,
						).Observe(time.Since(begin).Seconds())
				}(time.Now())
				resp, err := e(ctx, request)
				codeInt := detectCode(err)
				code = strconv.Itoa(codeInt)
				return resp, err
			}
		},
	)(svc, endpoints)
}

func (m serviceMetricMiddleware) newSubService(evalName string) subMetricMiddleware {
	return subMetricMiddleware{serviceMetricMiddleware: m, subServiceName: evalName}
}

type subMetricMiddleware struct {
	serviceMetricMiddleware
	subServiceName string
}

func (m subMetricMiddleware) New(svc middleware.Service, endpoints middleware.ServiceEndpoints) (middleware.Service, error) {
	return middleware.MakeMiddlewareFactory(nil,
		func(methodName string, service middleware.Service, defaultEndpoint func() endpoint.Endpoint) endpoint.Endpoint {
			e := defaultEndpoint()
			return func(ctx context.Context, request interface{}) (response interface{}, err error) {
				var code string
				defer func(begin time.Time) {
					m.subRequestCount.
						With(
							"sub_service", m.subServiceName,
							"service", m.serviceName,
						).Add(1)
					m.subResponseHistogram.
						With(
							"sub_service", m.subServiceName,
							"service", m.serviceName,
							"code", code,
						).Observe(time.Since(begin).Seconds())
				}(time.Now())
				resp, err := e(ctx, request)
				codeInt := detectCode(err)
				code = strconv.Itoa(codeInt)
				return resp, err
			}
		},
	)(svc, endpoints)
}
