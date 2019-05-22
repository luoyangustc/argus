package proxy_hook

import (
	"net/http"
)

type beforeRequestHook func(*http.Request) (httpStatusCode, errCode int,
	err error)

type afterRequestHook func(*http.Request, []byte) (header http.Header,
	newData []byte)

var (
	beforeRequests = make([]beforeRequestHook, 0, 3)
	afterRequests  = make([]afterRequestHook, 0, 3)
)

func registerBeforeRequest(h beforeRequestHook) {
	beforeRequests = append(beforeRequests, h)
}

func registerAfterRequest(h afterRequestHook) {
	afterRequests = append(afterRequests, h)
}

func CallBeforeRequest(req *http.Request) (httpStatusCode, errCode int,
	err error) {
	for _, h := range beforeRequests {
		httpStatusCode, errCode, err = h(req)
		if err != nil {
			return
		}
	}
	return
}

func CallAfterRequest(req *http.Request, data []byte) (header http.Header,
	newData []byte) {
	header, newData = http.Header{}, data
	for _, hook := range afterRequests {
		h, nd := hook(req, newData)
		newData = nd
		if h == nil {
			continue
		}
		for k, vs := range h {
			for _, v := range vs {
				header.Add(k, v)
			}
		}
	}
	return
}
