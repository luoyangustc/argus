package http

import (
	"context"
	"fmt"
	"net/http"
	"strconv"
	"strings"
)

func ParseRangeRequest(ctx context.Context, header http.Header) (*int64, *int64) {
	const key = "Range"
	const prefix = "bytes="
	value := header.Get(key)
	if value == "" {
		return nil, nil
	}
	if !strings.HasPrefix(value, prefix) { // 暂不支持其他形式的RangeRequest
		return nil, nil
	}
	value = value[6:]
	index := strings.Index(value, "-")
	if index == -1 {
		return nil, nil
	}
	begin, err := strconv.ParseInt(value[:index], 10, 64)
	if err != nil || begin < 0 { // 暂不支持负值
		return nil, nil
	}
	var end *int64
	if index < len(value)-1 {
		end0, err := strconv.ParseInt(value[index+1:], 10, 64)
		if err != nil || end0 < 0 || end0 < begin { // 暂不支持负值
			return nil, nil
		}
		end = &end0
	}
	return &begin, end
}

func FormatRangeRequest(header http.Header, beginOff, endOff *int64) {
	const key = "Range"

	var value string
	if beginOff != nil {
		value = fmt.Sprintf("bytes=%d-", *beginOff)
		if endOff != nil {
			value += strconv.FormatInt(*endOff, 10)
		}
	}
	if value != "" {
		header.Set(key, value)
	}
}

func FormatRangeResponse(header http.Header, begin, end, size int64) {
	const key = "Content-Range"
	header.Set(key, fmt.Sprintf("bytes %d-%d/%d", begin, end, size))
}
