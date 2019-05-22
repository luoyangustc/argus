package http

import (
	"context"
	"net/http"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestParseRangeRequest(t *testing.T) {

	begin, end := ParseRangeRequest(context.Background(), map[string][]string{})
	assert.Nil(t, begin)
	assert.Nil(t, end)

	begin, end = ParseRangeRequest(context.Background(), map[string][]string{"Range": {"bytes"}})
	assert.Nil(t, begin)
	assert.Nil(t, end)
	begin, end = ParseRangeRequest(context.Background(), map[string][]string{"Range": {"bytes=0"}})
	assert.Nil(t, begin)
	assert.Nil(t, end)
	begin, end = ParseRangeRequest(context.Background(), map[string][]string{"Range": {"bytes=-"}})
	assert.Nil(t, begin)
	assert.Nil(t, end)

	begin, end = ParseRangeRequest(context.Background(), map[string][]string{"Range": {"bytes=-1-0"}})
	assert.Nil(t, begin)
	assert.Nil(t, end)
	begin, end = ParseRangeRequest(context.Background(), map[string][]string{"Range": {"bytes=0--1"}})
	assert.Nil(t, begin)
	assert.Nil(t, end)

	begin, end = ParseRangeRequest(context.Background(), map[string][]string{"Range": {"bytes=0-0"}})
	assert.NotNil(t, begin)
	assert.Equal(t, int64(0), *begin)
	assert.NotNil(t, end)
	assert.Equal(t, int64(0), *end)

	begin, end = ParseRangeRequest(context.Background(), map[string][]string{"Range": {"bytes=1-"}})
	assert.NotNil(t, begin)
	assert.Equal(t, int64(1), *begin)
	assert.Nil(t, end)

}

func TestFormatRangeRequest(t *testing.T) {

	var off = func(off int64) *int64 { return &off }

	header := http.Header{}
	FormatRangeRequest(header, nil, nil)
	assert.Equal(t, "", header.Get("Range"))

	header = http.Header{}
	FormatRangeRequest(header, nil, off(1))
	assert.Equal(t, "", header.Get("Range"))

	header = http.Header{}
	FormatRangeRequest(header, off(0), off(1))
	assert.Equal(t, "bytes=0-1", header.Get("Range"))

	header = http.Header{}
	FormatRangeRequest(header, off(1), nil)
	assert.Equal(t, "bytes=1-", header.Get("Range"))

}

func TestFormatRangeResponse(t *testing.T) {

	header := http.Header{}
	FormatRangeResponse(header, 0, 1, 10)
	assert.Equal(t, "bytes 0-1/10", header.Get("Content-Range"))

}
