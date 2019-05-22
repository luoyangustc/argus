package eval

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"net/url"
	"os"
	"strconv"
	"strings"

	"github.com/prometheus/client_golang/prometheus"

	"github.com/qiniu/http/httputil.v1"
	"github.com/qiniu/xlog.v1"
	"qiniu.com/argus/atserving/model"
	STS "qiniu.com/argus/sts/client"
)

var (
	APP     string
	VERSION string

	NAMESPACE string = "ava"
	SUBSYSTEM string = "serving_eval"
)

func pStr(str string) *string { return &str }

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

//----------------------------------------------------------------------------//

var _ Stream = file{}

type file struct {
	uri string // file:///xx
}

// NewFile ...
func NewFile(uri string) Stream { return file{uri: uri} }

func (f file) Name() string { return f.uri }

func (f file) Open(ctx context.Context) (io.ReadCloser, int64, error) {
	xl := xlog.FromContextSafe(ctx)
	info, err := os.Lstat(strings.TrimPrefix(f.uri, _SchemeFilePrefix))
	if err != nil {
		xl.Warnf("get file stat failed. %s %v", f.uri, err)
		return nil, 0, err
	}
	_f, err := os.Open(strings.TrimPrefix(f.uri, _SchemeFilePrefix))
	if err != nil {
		xl.Warnf("open file failed. %s %v", f.uri, err)
	}
	return _f, info.Size(), err
}

func (f file) Clean() error {
	return os.Remove(strings.TrimPrefix(f.uri, _SchemeFilePrefix))
}

//----------------------------------------------------------------------------//

var _ Stream = mem{}

type mem struct {
	bs model.BYTES
}

func NewMem(bs model.BYTES) Stream { return mem{bs: bs} }

func (m mem) Name() string     { return "" }
func (m mem) Clean() error     { return nil }
func (m mem) GoString() string { return fmt.Sprintf("[]byte:%d", len(m.bs)) }

func (m mem) Open(ctx context.Context) (io.ReadCloser, int64, error) {
	return ioutil.NopCloser(bytes.NewReader(m.bs)), int64(len(m.bs)), nil
}

//----------------------------------------------------------------------------//

var _ Stream = &stsStream{}

type stsStream struct {
	STS.Client
	uri string
}

func newstsStream(client STS.Client, uri string) *stsStream {
	return &stsStream{
		Client: client,
		uri:    uri,
	}
}

func (s *stsStream) Name() string { return s.uri }
func (s *stsStream) Open(ctx context.Context) (io.ReadCloser, int64, error) {
	reader, length, _, err := s.Get(ctx, s.uri, nil)
	return reader, length, err
}
func (s *stsStream) Clean() error { return nil }

//----------------------------------------------------------------------------//

func formatError(err error) string {
	if err == nil {
		return ""
	}
	code, _ := httputil.DetectError(err)
	return strconv.Itoa(code)
}

func recoverAsError(r interface{}) error {
	switch x := r.(type) {
	case string:
		return errors.New(x)
	case error:
		return x
	default:
		return errors.New("Unknown panic")
	}
}

//----------------------------------------------------------------------------//

func responseTime() *prometheus.HistogramVec {
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
			ConstLabels: map[string]string{"app": APP, "version": VERSION},
		},
		[]string{"method", "error", "number"},
	)
	err := prometheus.Register(vec)
	if err != nil {
		if arError, ok := err.(prometheus.AlreadyRegisteredError); ok {
			return arError.ExistingCollector.(*prometheus.HistogramVec)
		}
		return nil
	}
	return vec
}

var _TaskAttempts = func() *prometheus.HistogramVec {
	vec := prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: NAMESPACE,
			Subsystem: SUBSYSTEM,
			Name:      "task_attempts",
			Help:      "nsq task attempt time",
			Buckets: []float64{
				0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
				10, 20, 30, 40, 50, 60, 70, 80, 90,
			},
			ConstLabels: map[string]string{"app": APP, "version": VERSION},
		},
		[]string{},
	)
	prometheus.MustRegister(vec)
	return vec
}()

func taskAttempts() *prometheus.HistogramVec { return _TaskAttempts }

var _RunBatchSize = func() *prometheus.HistogramVec {
	vec := prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: NAMESPACE,
			Subsystem: SUBSYSTEM,
			Name:      "run_batchsize",
			Help:      "evals run batchsize",
			Buckets: []float64{
				0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
				10, 20, 30, 40, 50,
			},
			ConstLabels: map[string]string{"app": APP, "version": VERSION},
		},
		[]string{},
	)
	prometheus.MustRegister(vec)
	return vec
}()

func runBatchSize() *prometheus.HistogramVec { return _RunBatchSize }
