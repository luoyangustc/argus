package notify_filter

import (
	"context"
	"encoding/base64"
	"fmt"
	"io"
	"net/http"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestNotifyFilter(t *testing.T) {

	{
		uid := uint32(123456)
		bucket := "bucket-test"
		key := "key-test"

		req := http.Request{
			Header: make(map[string][]string),
		}
		req.Header.Add("X-Qiniu-Uid", base64.StdEncoding.EncodeToString(
			[]byte(fmt.Sprintf("%d", uid)),
		))
		req.Header.Add("X-Qiniu-Bucket", base64.StdEncoding.EncodeToString(
			[]byte(bucket),
		))
		req.Header.Add("X-Qiniu-Key", base64.StdEncoding.EncodeToString(
			[]byte(key),
		))

		uidret, bucketret, keyret, err := reqSourceInfo(&req)
		assert.NoError(t, err)
		assert.Equal(t, uidret, uid)
		assert.Equal(t, bucketret, bucket)
		assert.Equal(t, keyret, key)
	}

	{
		mr := MockReader{Len: 10}
		bts, err := reqBody("", &mr)
		assert.NoError(t, err)
		assert.Equal(t, len(bts), 10)

		mr2 := MockReader{Len: 10}
		bts2, err := reqBody("====", &mr2)
		assert.Error(t, err)
		assert.Equal(t, len(bts2), 0)
	}

	{
		ctx := context.Background()
		s := NewMockService()
		_, err := s.PostHandler(ctx, &struct {
			ReqBody io.ReadCloser
			Cmd     string `json:"cmd"`
			URL     string `json:"url"`
		}{
			Cmd: "notify-filter/",
		}, nil)
		assert.Error(t, err)

		_, err = s.PostHandler(ctx, &struct {
			ReqBody io.ReadCloser
			Cmd     string `json:"cmd"`
			URL     string `json:"url"`
		}{
			Cmd: "notify-filter/true",
		}, nil)
		assert.Error(t, err)

		_, err = s.PostHandler(ctx, &struct {
			ReqBody io.ReadCloser
			Cmd     string `json:"cmd"`
			URL     string `json:"url"`
		}{
			Cmd: "notify-filter/true/xxxurl",
		}, nil)
		assert.NoError(t, err)
	}

}
