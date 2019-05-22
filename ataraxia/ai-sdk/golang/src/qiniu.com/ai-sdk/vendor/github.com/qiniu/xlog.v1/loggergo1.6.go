// +build !go1.7

package xlog

import (
	"net/http"
)

func New(w http.ResponseWriter, req *http.Request) *Logger {

	reqId := req.Header.Get(reqidKey)
	if reqId == "" {
		reqId = genReqId()
		req.Header.Set(reqidKey, reqId)
	}
	h := w.Header()
	h.Set(reqidKey, reqId)

	return &Logger{h, reqId, defaultCallDepth, nil}
}

func NewWithReq(req *http.Request) *Logger {

	reqId := req.Header.Get(reqidKey)
	if reqId == "" {
		reqId = genReqId()
		req.Header.Set(reqidKey, reqId)
	}
	h := http.Header{reqidKey: []string{reqId}}

	return &Logger{h, reqId, defaultCallDepth, nil}
}
