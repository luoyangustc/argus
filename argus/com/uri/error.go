package uri

import (
	"net/http"

	"github.com/qiniu/rpc.v3"
)

func responseError(resp *http.Response) (err error) {
	return rpc.ResponseError(resp)
}
