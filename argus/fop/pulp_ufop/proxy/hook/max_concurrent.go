package proxy_hook

import (
	"errors"
	"net/http"
	"sync/atomic"

	"qiniu.com/argus/fop/pulp_ufop/proxy/config"
)

var (
	reqCount uint32 = 0
)

func incCurrentRequestCount() {
	atomic.AddUint32(&reqCount, 1)
}

func decCurrentRequestCount() {
	atomic.AddUint32(&reqCount, ^uint32(0))
}

func init() {
	registerBeforeRequest(beforeRequestHook(func(*http.Request) (
		httpStatusCode, errCode int, err error) {
		if atomic.LoadUint32(&reqCount) > proxy_config.MaxConcurrent() {
			return http.StatusTooManyRequests, http.StatusTooManyRequests,
				errors.New("server is busy, please try later")
		}
		incCurrentRequestCount()
		return 0, 0, nil
	}))
	registerAfterRequest(afterRequestHook(func(req *http.Request,
		data []byte) (http.Header, []byte) {
		decCurrentRequestCount()
		return nil, data
	}))
}
