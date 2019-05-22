package monitor

import (
	"context"
	"crypto/tls"
	"io"
	"net"
	"net/http"
	"net/http/httptrace"
	"strconv"
	"time"

	"github.com/prometheus/client_golang/prometheus"

	"github.com/qiniu/xlog.v1"
)

const (
	namespace = "argus"
	subsystem = "monitor"
)

var _ReqTime *prometheus.HistogramVec
var _ReqOverdue *prometheus.CounterVec
var _ReqError *prometheus.CounterVec

func init() {
	_ReqTime = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: namespace,
			Subsystem: subsystem,
			Name:      "request_time",
			Help:      "request time of requests",
			Buckets: []float64{0,
				50, 100, 150, 200, 250, 300, 350, 400, 450, 500,
				600, 700, 800, 900, 1000,
				2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000,
				10000, 20000, 30000, 40000, 50000, 60000}, // millisecond
		}, []string{"host", "api", "stage"})

	_ReqOverdue = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: namespace,
			Subsystem: subsystem,
			Name:      "request_overdue",
			Help:      "overdue requests",
		}, []string{"host", "api"})

	_ReqError = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: namespace,
			Subsystem: subsystem,
			Name:      "request_error",
			Help:      "error requests",
		}, []string{"host", "api", "error"})

	prometheus.MustRegister(_ReqTime, _ReqOverdue, _ReqError)
}

func MonitorHTTP(ctx context.Context, req *http.Request,
	host, api string, timeout time.Duration,
) {

	xl := xlog.FromContextSafe(ctx)

	var t0, t1, t2, t3, t4, t5, t6 time.Time
	_, _, _ = t2, t5, t6

	trace := &httptrace.ClientTrace{
		DNSStart: func(_ httptrace.DNSStartInfo) { t0 = time.Now() },
		DNSDone:  func(_ httptrace.DNSDoneInfo) { t1 = time.Now() },
		ConnectStart: func(_, _ string) {
			if t1.IsZero() {
				// connecting to IP
				t1 = time.Now()
			}
		},
		ConnectDone: func(net, addr string, err error) {
			if err != nil {
				xl.Errorf("unable to connect to host %v: %v", addr, err)
			}
			t2 = time.Now()
		},
		GotConn:              func(_ httptrace.GotConnInfo) { t3 = time.Now() },
		GotFirstResponseByte: func() { t4 = time.Now() },
		TLSHandshakeStart:    func() { t5 = time.Now() },
		TLSHandshakeDone:     func(_ tls.ConnectionState, _ error) { t6 = time.Now() },
	}
	req = req.WithContext(httptrace.WithClientTrace(context.Background(), trace))

	client := &http.Client{
		Transport: &http.Transport{
			Proxy: http.ProxyFromEnvironment,
			DialContext: (&net.Dialer{
				Timeout:   5 * time.Second,
				KeepAlive: 30 * time.Second,
				DualStack: true,
			}).DialContext,
			MaxIdleConns:          100,
			IdleConnTimeout:       90 * time.Second,
			TLSHandshakeTimeout:   10 * time.Second,
			ExpectContinueTimeout: 1 * time.Second,
		},
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			// always refuse to follow redirects, visit does that
			// manually if required.
			return http.ErrUseLastResponse
		},
		Timeout: time.Second * 60,
	}

	resp, err := client.Do(req)
	if err != nil {
		xl.Errorf("failed to read response: %v", err)
		_ReqError.WithLabelValues(host, api, err.Error()).Inc()
	}

	if resp.Body != nil {
		_, _ = readAll(resp.Body)
		resp.Body.Close()
	}

	t7 := time.Now() // after read body
	if t0.IsZero() {
		// we skipped DNS
		t0 = t1
	}

	if resp.StatusCode >= 400 {
		_ReqError.WithLabelValues(host, api, strconv.Itoa(resp.StatusCode)).Inc()
	}
	if t7.Sub(t0) >= timeout {
		_ReqOverdue.WithLabelValues(host, api).Inc()
	}

	_ReqTime.WithLabelValues(host, api, "dns").Observe(float64(t1.Sub(t0)) / float64(time.Millisecond))
	_ReqTime.WithLabelValues(host, api, "tcp").Observe(float64(t3.Sub(t0)) / float64(time.Millisecond))
	_ReqTime.WithLabelValues(host, api, "server").Observe(float64(t4.Sub(t3)) / float64(time.Millisecond))
	_ReqTime.WithLabelValues(host, api, "content").Observe(float64(t7.Sub(t4)) / float64(time.Millisecond))
	_ReqTime.WithLabelValues(host, api, "total").Observe(float64(t7.Sub(t0)) / float64(time.Millisecond))

	xl.Infof("%s %d", req.URL.String(), resp.StatusCode)

}

func readAll(r io.Reader) (m int, err error) {
	var buf = make([]byte, 1024*16)
	for {
		n, er := r.Read(buf)
		m += n
		if er != nil {
			if er != io.EOF {
				err = er
			}
			break
		}
	}
	return
}
