package video

import (
	"context"
	"time"

	"github.com/go-kit/kit/endpoint"
	"github.com/go-kit/kit/metrics"
	kitprometheus "github.com/go-kit/kit/metrics/prometheus"
	"github.com/prometheus/client_golang/prometheus"
	"qiniu.com/argus/service/middleware"
	"qiniu.com/argus/service/scenario/video/dashboard"
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
	requestProcessingGauge metrics.Gauge
	responseHistogram      metrics.Histogram

	opResponseHistogram   metrics.Histogram
	evalResponseHistogram metrics.Histogram
	// logger log.Logger
}

func newMetricMiddleware(conf MetricConfig) *_MetricMiddleware {
	if len(conf.Namespace) == 0 {
		conf.Namespace = "qiniu_ai"
	}
	if len(conf.Subsystem) == 0 {
		conf.Subsystem = "video"
	}

	conf.Dashborads = dashboard.Load()

	return &_MetricMiddleware{
		MetricConfig: conf,

		requestCount: kitprometheus.NewCounterFrom(
			prometheus.CounterOpts{
				Namespace: conf.Namespace,
				Subsystem: conf.Subsystem,
				Name:      "request_count",
				Help:      "Number of requests received.",
			},
			[]string{"format"},
		),
		requestProcessingGauge: kitprometheus.NewGaugeFrom(
			prometheus.GaugeOpts{
				Namespace: conf.Namespace,
				Subsystem: conf.Subsystem,
				Name:      "request_processing",
				Help:      "Number of request processing",
			},
			[]string{"format"},
		),
		responseHistogram: kitprometheus.NewHistogramFrom(
			prometheus.HistogramOpts{
				Namespace: conf.Namespace,
				Subsystem: conf.Subsystem,
				Name:      "response_time",
				Help:      "response time of requests",
				Buckets: []float64{
					0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
					10, 20, 30, 40, 50,
					60, 120, 180, 240, 300, 360, 420, 480, 540,
					600, 1200, 1800, 2400, 3000, 3600,
				}, // time.second
			},
			[]string{"format"},
		),

		opResponseHistogram: kitprometheus.NewHistogramFrom(
			prometheus.HistogramOpts{
				Namespace: conf.Namespace,
				Subsystem: conf.Subsystem,
				Name:      "op_response_time",
				Help:      "response time of op requests",
				Buckets: []float64{
					0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
					1, 2, 3, 4, 5, 6, 7, 8, 9,
					10, 20, 30, 40, 50,
				}, // time.second
			},
			[]string{"op"},
		),

		evalResponseHistogram: kitprometheus.NewHistogramFrom(
			prometheus.HistogramOpts{
				Namespace: conf.Namespace,
				Subsystem: conf.Subsystem,
				Name:      "eval_response_time",
				Help:      "response time of eval requests",
				Buckets: []float64{
					0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
					1, 2, 3, 4, 5, 6, 7, 8, 9,
					10, 20, 30, 40, 50,
				}, // time.second
			},
			[]string{"eval"},
		),
	}
}

var _ endpoint.Middleware = (&_MetricMiddleware{}).Video

func (m *_MetricMiddleware) Video(ep endpoint.Endpoint) endpoint.Endpoint {
	return func(ctx context.Context, req interface{}) (interface{}, error) {
		m.requestCount.With("format", "").Add(1)
		m.requestProcessingGauge.With("format", "").Add(1)
		defer func(begin time.Time) {
			m.requestProcessingGauge.With("format", "").Add(-1)
			m.responseHistogram.With("format", "").Observe(time.Since(begin).Seconds())
		}(time.Now())
		return ep(ctx, req)
	}
}

func (m *_MetricMiddleware) Eval(op string) endpoint.Middleware {
	return func(ep endpoint.Endpoint) endpoint.Endpoint {
		return func(ctx context.Context, req interface{}) (interface{}, error) {
			defer func(begin time.Time) {
				m.opResponseHistogram.With("op", op).Observe(time.Since(begin).Seconds())
			}(time.Now())
			return ep(ctx, req)
		}
	}
}

func (m *_MetricMiddleware) NewEvalService(name string) evalMetricMiddleware {
	return evalMetricMiddleware{_MetricMiddleware: m, evalName: name}
}

type evalMetricMiddleware struct {
	*_MetricMiddleware
	evalName string
}

func (m evalMetricMiddleware) New(svc middleware.Service, endpoints middleware.ServiceEndpoints) (middleware.Service, error) {
	return middleware.MakeMiddlewareFactory(nil,
		func(methodName string, service middleware.Service, defaultEndpoint func() endpoint.Endpoint) endpoint.Endpoint {
			e := defaultEndpoint()
			return func(ctx context.Context, request interface{}) (response interface{}, err error) {
				defer func(begin time.Time) {
					m.evalResponseHistogram.With("eval", m.evalName).Observe(time.Since(begin).Seconds())
				}(time.Now())
				return e(ctx, request)
			}
		},
	)(svc, endpoints)
}
