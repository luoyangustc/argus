package misc

import (
	"strconv"

	"github.com/prometheus/client_golang/prometheus"
)

const (
	NAMESPACE = "argus"
	SUBSYSTEM = "ccp_review"
)

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
		[]string{"api", "code"},
	)
	prometheus.MustRegister(vec)
	return vec
}()

func ResponseTime(api string, code int) prometheus.Histogram {
	return _ResponseTime.WithLabelValues(api, strconv.Itoa(code))
}

var _RequestsCounter = func() *prometheus.CounterVec {
	vec := prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: NAMESPACE,
			Subsystem: SUBSYSTEM,
			Name:      "requests_counter",
			Help:      "number of requests",
		},
		[]string{"api", "code"},
	)
	prometheus.MustRegister(vec)
	return vec
}()

func RequestsCounter(api string, code int) prometheus.Counter {
	return _RequestsCounter.WithLabelValues(api, strconv.Itoa(code))
}
