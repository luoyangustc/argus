package segment

import (
	"context"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	xlog "github.com/qiniu/xlog.v1"
)

var (
	NAMESPACE = "ava"
	SUBSYSTEM = "argus_segment"
)

func durationAsFloat64(d time.Duration) float64 {
	return float64(d/time.Millisecond) / 1000
}

var _ResponseTimeShort = func() *prometheus.HistogramVec {
	vec := prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: NAMESPACE,
			Subsystem: SUBSYSTEM,
			Name:      "response_time_short",
			Help:      "Short Response time of requests",
			Buckets: []float64{
				0,
				0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
				0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
				1, 2, 3, 4, 5, 6, 7, 8, 9,
			}, // time.second
		},
		[]string{"method", "item", "code"},
	)
	prometheus.MustRegister(vec)
	return vec
}()

func responseTimeShort(op, method, item, code string) prometheus.Histogram {
	return _ResponseTimeShort.WithLabelValues(op, method, item, code)
}

var _ResponseTimeLong = func() *prometheus.HistogramVec {
	vec := prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: NAMESPACE,
			Subsystem: SUBSYSTEM,
			Name:      "response_time_long",
			Help:      "Long Response time of requests",
			Buckets: []float64{
				0,
				1, 2, 3, 4, 5, 6, 7, 8, 9,
				10, 20, 30, 40, 50, 60,
				120, 180, 240, 300, 360, 420, 480, 540, 600,
			}, // time.second
		},
		[]string{"method", "item", "code"},
	)
	prometheus.MustRegister(vec)
	return vec
}()

func responseTimeLong(method, item, code string) prometheus.Histogram {
	return _ResponseTimeLong.WithLabelValues(method, item, code)
}

var _RequestsCounter = func() *prometheus.CounterVec {
	vec := prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: NAMESPACE,
			Subsystem: SUBSYSTEM,
			Name:      "requests_counter",
			Help:      "number of requests",
		},
		[]string{"method", "item", "code"},
	)
	prometheus.MustRegister(vec)
	return vec
}()

func requestsCounter(method, item, code string) prometheus.Counter {
	return _RequestsCounter.WithLabelValues(method, item, code)
}

var _RequestsParallel = func() *prometheus.GaugeVec {
	vec := prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Namespace: NAMESPACE,
			Subsystem: SUBSYSTEM,
			Name:      "requests_parallel",
			Help:      "parallel of requests",
		},
		[]string{"method"},
	)
	prometheus.MustRegister(vec)
	return vec
}()

func requestsParallel(method string) prometheus.Gauge {
	return _RequestsParallel.WithLabelValues(method)
}

////////////////////////////////////////////////////////////////////////////////

func spawnContext(ctx context.Context) context.Context {
	return xlog.NewContext(ctx, xlog.FromContextSafe(ctx).Spawn())
}
