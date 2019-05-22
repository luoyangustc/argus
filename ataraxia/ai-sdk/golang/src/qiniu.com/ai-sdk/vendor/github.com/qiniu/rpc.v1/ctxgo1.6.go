// +build !go1.7

package rpc

import "net/http"

// go1.6及以下不作实际处理
func (r *Client) doCtx(l Logger, req *http.Request) (resp *http.Response, err error) {

	return r.Client.Do(req)
}
