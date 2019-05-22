package gate

import "github.com/prometheus/client_golang/prometheus"

var logPushResponseTime = prometheus.NewHistogramVec(
	prometheus.HistogramOpts{
		Namespace: NAMESPACE,
		Subsystem: SUBSYSTEM,
		Name:      "log_push_response_time",
		Help:      "Response time of requests",
		Buckets: []float64{
			0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
			1, 2, 3, 4, 5, 6, 7, 8, 9,
			10, 20, 30, 40, 50, 60,
		}, // time.second
	},
	[]string{"status"},
)

// 注意目前实现，只能统计大小小于 maxLogPushFileSize 的文件
var resourceSize = prometheus.NewHistogramVec(
	prometheus.HistogramOpts{
		Namespace: NAMESPACE,
		Subsystem: SUBSYSTEM,
		Name:      "resource_size",
		Help:      "处理的图片大小统计",
		Buckets: []float64{
			1e3, 1e4, 1e5, 1e6, // 1kb -> 1mb
			2 * 1e6,  // 2mb
			4 * 1e6,  // 4mb
			8 * 1e6,  // 8mb
			16 * 1e6, // 16mb
			32 * 1e6, // 32mb
			64 * 1e6, // 64mb
		},
	},
	[]string{"cmd"},
)

var logPushSendBytes = prometheus.NewCounter(
	prometheus.CounterOpts{
		Namespace: NAMESPACE,
		Subsystem: SUBSYSTEM,
		Name:      "log_push_send_bytes",
		Help:      "LogPushClient 发送的 bytes 数目",
	},
)

var logPushSkipFile = prometheus.NewCounter(
	prometheus.CounterOpts{
		Namespace: NAMESPACE,
		Subsystem: SUBSYSTEM,
		Name:      "log_push_skip_file_num",
		Help:      "LogPushClient 由于文件太大跳过的文件数目",
	},
)

var reqProcessing = prometheus.NewGauge(prometheus.GaugeOpts{
	Namespace: NAMESPACE,
	Subsystem: SUBSYSTEM,
	Name:      "requests_processing",
	Help:      "Number of processing requests",
})

var logPushMemory = prometheus.NewGauge(prometheus.GaugeOpts{
	Namespace: NAMESPACE,
	Subsystem: SUBSYSTEM,
	Name:      "log_push_memory",
	Help:      "LogPushClient 占用的内存",
})

var logPushProcessing = prometheus.NewGauge(prometheus.GaugeOpts{
	Namespace: NAMESPACE,
	Subsystem: SUBSYSTEM,
	Name:      "log_push_processing",
	Help:      "Number of processing log push requests",
})

func init() {
	prometheus.MustRegister(logPushResponseTime, logPushSendBytes, logPushSkipFile, reqProcessing, logPushMemory, logPushProcessing, resourceSize)
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
		[]string{"cmd", "version", "method", "error", "number"},
	)
	prometheus.MustRegister(vec)
	return vec
}()

func responseTime() *prometheus.HistogramVec { return _ResponseTime }

var _RequestsCounter = func() *prometheus.CounterVec {
	vec := prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: NAMESPACE,
			Subsystem: SUBSYSTEM,
			Name:      "requests_counter",
			Help:      "number of requests",
		},
		[]string{"cmd", "method", "code"},
	)
	prometheus.MustRegister(vec)
	return vec
}()

func requestsCounter(cmd, method, code string) prometheus.Counter {
	return _RequestsCounter.WithLabelValues(cmd, method, code)
}
