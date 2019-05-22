package sts

import (
	"context"
	"time"

	"github.com/qiniu/xlog.v1"
	URIC "qiniu.com/argus/com/uri"
	URI "qiniu.com/argus/sts/uri"
)

func needFetch(uri URI.Uri) bool {
	switch URI.TypeOf(uri) {
	case URI.HTTP, URI.QINIU:
		return true
	default:
		return false
	}
}

type Fetcher interface {
	Fetch(context.Context, URI.Uri, *int64, File) error
	Get(context.Context, URI.Uri, ...URIC.GetOption) (*URIC.Response, error)
}

var _ Fetcher = fetcher{}

type fetcher struct {
	URIC.Handler
}

func NewFetcher(handler URIC.Handler) fetcher { return fetcher{Handler: handler} }

func (f fetcher) Fetch(ctx context.Context, uri URI.Uri, length *int64, file File) error {
	// TODO 超时
	xl := xlog.FromContextSafe(ctx)
	resp, err := f.Handler.Get(ctx, URIC.Request{URI: uri.ToString()})
	if err != nil {
		xl.Warnf("fetch failed. %s %s", uri, err)
		_FetchFaileds.WithLabelValues(URI.SchemeOf(uri), formatError(err)).Inc()
		return ErrFetchFailed
	}
	defer resp.Body.Close()
	var n int64
	if length == nil {
		file.SetLength(resp.Size)
	}
	defer func(begin time.Time) {
		d := time.Since(begin)
		xl.Infof("fetch file done. %d %s %v", n, d, err)
		responseTime().
			WithLabelValues("io.Fetch", formatError(err)).
			Observe(float64(d) / 1e9)
		_IORate.
			WithLabelValues("io.Fetch", URI.SchemeOf(uri), formatError(err)).
			Observe(float64(n) / (float64(n) / 1e9))
	}(time.Now())
	n, err = file.Write(ctx, resp.Body)
	return err
}

func (f fetcher) Get(ctx context.Context, uri URI.Uri, opts ...URIC.GetOption) (*URIC.Response, error) {
	// TODO 超时
	xl := xlog.FromContextSafe(ctx)
	resp, err := f.Handler.Get(ctx, URIC.Request{URI: uri.ToString()}, opts...)
	if err != nil {
		xl.Warnf("fetch failed. %s %s", uri, err)
		_FetchFaileds.WithLabelValues(URI.SchemeOf(uri), formatError(err)).Inc()
		return nil, ErrFetchFailed
	}
	return resp, nil
}
