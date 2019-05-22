package utility

import (
	"context"
	"crypto/sha1"
	"encoding/base64"
	"encoding/hex"
	"fmt"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/qiniu/http/httputil.v1"

	ahttp "qiniu.com/argus/argus/com/http"
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

func improveURI(uri string, uid uint32) (string, error) {
	_url, err := url.Parse(uri)
	if err != nil {
		return uri, err
	}
	if _url.Scheme != "qiniu" {
		return uri, nil
	}
	_url.User = url.User(strconv.Itoa(int(uid)))
	return _url.String(), nil
}

////////////////////////////////////////////////////////////////////////////////

func callRetry(ctx context.Context, f func(context.Context) error) error {
	return ahttp.CallWithRetry(ctx, []int{530}, []func(context.Context) error{f, f})
}

////////////////////////////////////////////////////////////////////////////////

const (
	_DataURIPrefix = "data:application/octet-stream;base64,"
)

func dataURIEtag(uri string) (string, error) {
	data, err := base64.StdEncoding.DecodeString(strings.TrimPrefix(uri, _DataURIPrefix))
	if err != nil {
		return "", err
	}
	h := sha1.New()
	h.Write(data)
	return hex.Dump(h.Sum(nil)), nil
}

////////////////////////////////////////////////////////////////////////////////

func formatError(err error) string {
	if err == nil {
		return strconv.Itoa(200)
	}
	return strconv.Itoa(httputil.DetectCode(err))
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
	// prometheus.MustRegister(vec)
	return vec
}()

func responseTime() *prometheus.HistogramVec { return _ResponseTime }

func responseTimeAtServer(api, err string) prometheus.Histogram {
	return _ResponseTime.WithLabelValues(api, "S", "", err)
}

func responseTimeAtClient(api, item, err string) prometheus.Histogram {
	return _ResponseTime.WithLabelValues(api, "C", "item", err)
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
	// prometheus.MustRegister(vec)
	return vec
}()

func httpRequestsCounter(api, method, err string) prometheus.Counter {
	return _RequestsCounter.WithLabelValues(api, method, err)
}
