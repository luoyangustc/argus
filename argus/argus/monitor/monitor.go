package monitor

import (
	"net/http"
	"strconv"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"qiniupkg.com/http/httputil.v2"
)

const (
	namespace = "ava"
	subsystem = "argus_gate"
)

var m struct {
	respTime          *prometheus.HistogramVec
	inferenceRespTime *prometheus.HistogramVec
	reg               *prometheus.Registry
}

func init() {
	m.respTime = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Namespace: namespace,
		Subsystem: subsystem,
		Name:      "response_time",
		Help:      "Response time of requests",
		Buckets:   []float64{0, 100, 200, 300, 500, 1000, 5000, 10000, 20000, 30000, 60000}, // millisecond
	}, []string{"api", "code"})
	m.inferenceRespTime = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Namespace: namespace,
		Subsystem: subsystem,
		Name:      "inference_response_time",
		Help:      "Response time of inference API requests",
		Buckets:   []float64{0, 100, 200, 300, 500, 1000, 5000, 10000, 20000, 30000, 60000}, // millisecond
	}, []string{"api", "code"})
	reg := prometheus.NewRegistry()
	reg.MustRegister(m.respTime, m.inferenceRespTime)
	m.reg = reg
}

func Handler() http.Handler {
	return promhttp.HandlerFor(m.reg, promhttp.HandlerOpts{})
}

func ResponseTime(api string, err error, responseTime time.Duration) {
	m.respTime.WithLabelValues(api, strconv.Itoa(detectCode(err))).Observe(responseTime.Seconds() * 1e3)
}

func InferenceResponseTime(api string, err error, responseTime time.Duration) {
	m.inferenceRespTime.WithLabelValues(api, strconv.Itoa(detectCode(err))).Observe(responseTime.Seconds() * 1e3)
}

func detectCode(err error) int {
	if err == nil {
		return 200
	}
	return httputil.DetectCode(err)
}
