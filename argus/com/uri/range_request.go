package uri

import (
	"context"
	"fmt"
	"math"
	"net/http"

	HTTP "qiniu.com/argus/com/http"
)

// n, MaxInt64	=> n-
// n, m			=> n-m
func WithRange(begin, end int64) GetOption {
	return func(req *Request) {
		req.beginOff = &begin
		if end != math.MaxInt64 {
			req.endOff = &end
		} else {
			req.endOff = nil
		}
	}
}

func ParseRangeRequest(ctx context.Context, header http.Header) GetOption {
	begin, end := HTTP.ParseRangeRequest(ctx, header)
	if begin == nil {
		return nil
	}
	return func(req *Request) { req.beginOff, req.endOff = begin, end }
}

func FormatRangeRequest(header http.Header, req Request) {
	HTTP.FormatRangeRequest(header, req.beginOff, req.endOff)
}

func FormatRangeResponse(header http.Header, begin, end, size int64) {
	const key = "Content-Range"
	header.Set(key, fmt.Sprintf("bytes %d-%d/%d", begin, end, size))
}
