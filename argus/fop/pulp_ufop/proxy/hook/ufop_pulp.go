package proxy_hook

import (
	"net/http"

	"qiniu.com/argus/fop/pulp_ufop/proxy/cmd"
)

const ufopPulpsCmd string = cmd.UfopPulps

func init() {
	registerAfterRequest(afterRequestHook(
		func(req *http.Request, data []byte) (http.Header, []byte) {
			return pulp(req, data, ufopPulpsCmd)
		}))
}
