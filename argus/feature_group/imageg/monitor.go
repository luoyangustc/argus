package imageg

import (
	"strconv"

	"github.com/prometheus/client_golang/prometheus"
)

var (
	NAMESPACE = "argus"
	SUBSYSTEM = "ig"

	_RequestGaugeVec = func() *prometheus.GaugeVec {
		vec := prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Namespace: NAMESPACE,
				Subsystem: SUBSYSTEM,
				Name:      "request",
				Help:      "request count",
			},
			[]string{"api"},
		)
		prometheus.MustRegister(vec)
		return vec
	}()
	_RequestGauge = func(api string) prometheus.Gauge {
		return _RequestGaugeVec.WithLabelValues(api)
	}

	_ResponseTimeHistogramVec = func() *prometheus.HistogramVec {
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
	_ResponseTimeHistogram = func(api string, code int) prometheus.Histogram {
		return _ResponseTimeHistogramVec.WithLabelValues(api, strconv.Itoa(code))
	}

	_ClientTimeHistogramVec = func() *prometheus.HistogramVec {
		vec := prometheus.NewHistogramVec(
			prometheus.HistogramOpts{
				Namespace: NAMESPACE,
				Subsystem: SUBSYSTEM,
				Name:      "client_time",
				Help:      "client request time",
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
	_ClientTimeHistogram = func(api string, code int) prometheus.Histogram {
		return _ClientTimeHistogramVec.WithLabelValues(api, strconv.Itoa(code))
	}
)
