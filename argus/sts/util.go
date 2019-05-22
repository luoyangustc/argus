package sts

import (
	"strconv"

	"github.com/prometheus/client_golang/prometheus"

	"github.com/qiniu/http/httputil.v1"
)

func init() {
	prometheus.MustRegister(_OverdueOpens)
	prometheus.MustRegister(_FetchFaileds)
	prometheus.MustRegister(_IORate)
}

var (
	_NameSpace = "ava"
	_SubSystem = "serving_sts"
)

var (
	_OverdueOpens = prometheus.NewCounter(
		prometheus.CounterOpts{
			Namespace: _NameSpace,
			Subsystem: _SubSystem,
			Name:      "overdue_open",
			Help:      "Number of overdue opens.",
		},
	)

	_FetchFaileds = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: _NameSpace,
			Subsystem: _SubSystem,
			Name:      "fetch_failed",
			Help:      "Number of fetch faileds.",
		},
		[]string{"scheme", "error"},
	)

	_IORate = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: _NameSpace,
			Subsystem: _SubSystem,
			Name:      "io_rate",
			Help:      "io rate.",
			Buckets: []float64{
				0,
				1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
				1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072,
			}, // KB/s
		},
		[]string{"method", "type", "error"},
	)
)

func formatError(err error) string {
	if err == nil {
		return ""
	}
	code, _ := httputil.DetectError(err)
	return strconv.Itoa(code)
}

var _ResponseTime = func() *prometheus.HistogramVec {
	vec := prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: _NameSpace,
			Subsystem: _SubSystem,
			Name:      "response_time",
			Help:      "Response time of requests",
			Buckets: []float64{
				0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
				1, 2, 3, 4, 5, 6, 7, 8, 9,
				10, 20, 30, 40, 50, 60,
			}, // time.second
		},
		[]string{"method", "error"},
	)
	prometheus.MustRegister(vec)
	return vec
}()

func responseTime() *prometheus.HistogramVec { return _ResponseTime }

var _WaitForLock = func() *prometheus.GaugeVec {
	vec := prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Namespace: _NameSpace,
			Subsystem: _SubSystem,
			Name:      "wait_for_lock",
			Help:      "wait for lock",
		},
		[]string{"object", "lock", "method"},
	)
	prometheus.MustRegister(vec)
	return vec
}()

func waitForLock(object, lock, method string) prometheus.Gauge {
	return _WaitForLock.WithLabelValues(object, lock, method)
}
