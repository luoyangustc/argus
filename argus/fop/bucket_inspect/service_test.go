package bucket_inspect

import (
	"context"
	"encoding/base64"
	"fmt"
	"net/http"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestBktIsp(t *testing.T) {

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

		uidret, err := reqUID(&req)
		assert.NoError(t, err)
		assert.Equal(t, uidret, uid)
	}

	{
		ak, sk, err := aksk2(nil, 0)
		assert.Error(t, err)
		assert.Equal(t, ak, "")
		assert.Equal(t, sk, "")
	}

	{
		bts, err := reqBody("url", nil)
		assert.Error(t, err)
		assert.Nil(t, bts)
	}

	{
		s := Service{}
		err := s.disable(context.Background(), nil)
		assert.Error(t, err)
	}
}
