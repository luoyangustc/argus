package proxy

import (
	"context"
	"encoding/base64"
	"fmt"
	"net/http"
	"testing"

	restrpc "github.com/qiniu/http/restrpc.v1"
	"github.com/stretchr/testify/assert"
	"qiniu.com/argus/com/proxy/fop"
)

func TestBktIsp(t *testing.T) {

	{
		obj := struct {
			Fa string      `json:"fa"`
			Fb bool        `json:"fb"`
			Fc int         `json:"fc"`
			Fd interface{} `json:"fd"`
		}{
			Fa: "fa",
			Fb: true,
			Fc: 1,
		}
		assert.NotEmpty(t, JsonStr(obj))
	}
	{
		ic := NewImageCensor("_url")
		assert.NotNil(t, ic)

		_, err := ic.Post(context.Background(), fop.ProxyReq{}, &restrpc.Env{
			Req: &http.Request{},
		})
		assert.Error(t, err)
	}

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
		vc := NewVideoCensor("_url")
		assert.NotNil(t, vc)

		_, err := vc.Post(context.Background(), fop.ProxyReq{}, &restrpc.Env{
			Req: &http.Request{},
		})
		assert.Error(t, err)
	}
}
