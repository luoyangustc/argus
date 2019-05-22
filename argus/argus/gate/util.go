package gate

import (
	"fmt"
	"net/http"
	"time"

	"github.com/prometheus/client_golang/prometheus"
)

func setStateHeader(header http.Header, name string, state int) {
	const key = "X-Origin-A"

	if _, ok := header[key]; ok {
		header.Set(key, header.Get(key)+";"+fmt.Sprintf("%s:%d", name, state))
	} else {
		header.Set(key, fmt.Sprintf("%s:%d", name, state))
	}
}

////////////////////////////////////////////////////////////////////////////////

var (
	NAMESPACE = "ava"
	SUBSYSTEM = "argus_gate"
)

func durationAsFloat64(d time.Duration) float64 {
	return float64(d/time.Millisecond) / 1000
}

var _ResponseTime = func() *prometheus.HistogramVec {
	vec := prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: NAMESPACE,
			Subsystem: SUBSYSTEM,
			Name:      "response_time",
			Help:      "Response time of requests",
			Buckets: []float64{
				0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
				1, 2, 3, 4, 5, 6, 7, 8, 9,
				10, 20, 30, 40, 50, 60,
			}, // time.second
		},
		[]string{"api", "method", "item", "error"},
	)
	prometheus.MustRegister(vec)
	return vec
}()

func responseTime() *prometheus.HistogramVec { return _ResponseTime }

func responseTimeAtProxy(api, err string) prometheus.Histogram {
	return _ResponseTime.WithLabelValues(api, "P", "", err)
}

var _RequestsCounter = func() *prometheus.CounterVec {
	vec := prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: NAMESPACE,
			Subsystem: SUBSYSTEM,
			Name:      "http_requests_counter",
			Help:      "number of http requests",
		},
		[]string{"api", "method", "code"},
	)
	prometheus.MustRegister(vec)
	return vec
}()

func httpRequestsCounterAtProxy(api, err string) prometheus.Counter {
	return _RequestsCounter.WithLabelValues(api, "P", err)
}

var _RequestsParallel = func() *prometheus.GaugeVec {
	vec := prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Namespace: NAMESPACE,
			Subsystem: SUBSYSTEM,
			Name:      "requests_parallel",
			Help:      "parallel of requests",
		},
		[]string{"api", "method"},
	)
	prometheus.MustRegister(vec)
	return vec
}()

func requestsParallelAtProxy(api string) prometheus.Gauge {
	return _RequestsParallel.WithLabelValues(api, "P")
}
