package proxy

import (
	"context"
	"github.com/stretchr/testify/assert"
	"net/http"
	"testing"

	"github.com/qiniu/http/restrpc.v1"
	"qiniu.com/argus/com/proxy/fop"
)

func TestPulp(t *testing.T) {

	{

		pp := NewPulpv0Proxy("_url")
		assert.NotNil(t, pp)

		_, err := pp.Post(context.Background(), fop.ProxyReq{}, &restrpc.Env{
			Req: &http.Request{},
		})
		assert.Error(t, err)
	}
}
