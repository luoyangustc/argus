package teapot

import (
	"bufio"
	"fmt"
	"net"
	"net/http"
	"sync"
)

// ResponseWriter is a wrapper around http.ResponseWriter that provides extra information about
// the response. It is recommended that middleware handlers use this construct to wrap a responsewriter
// if the functionality calls for it.
type ResponseWriter interface {
	http.ResponseWriter
	http.Flusher
	http.Hijacker
	http.CloseNotifier
	// Status returns the status code of the response or 0 if the response has not been written.
	Status() int
	// Written returns whether or not the ResponseWriter has been written.
	Written() bool
	// Size returns the size of the response body.
	Size() int
	// Before allows for a function to be called before the ResponseWriter has been written to. This is
	// useful for setting headers or any other operations that must happen before a response has been written.
	Before(BeforeFunc)
}

// BeforeFunc is a function that is called before the ResponseWriter has been written to.
type BeforeFunc func(ResponseWriter)

// newResponseWriter creates a ResponseWriter that wraps an http.ResponseWriter
func newResponseWriter(rw http.ResponseWriter) ResponseWriter {
	return &responseWriter{
		ResponseWriter: rw,
		status:         0,
		size:           0,
		beforeFuncs:    nil,
	}
}

type responseWriter struct {
	http.ResponseWriter
	status      int
	size        int
	beforeFuncs []BeforeFunc

	once sync.Once

	hijacked bool
}

// not really flush header, just save status code
func (rw *responseWriter) WriteHeader(s int) {
	if s == 0 {
		panic("http status code can not be zero")
	}
	rw.status = s
}

func (rw *responseWriter) Write(b []byte) (int, error) {
	rw.once.Do(func() {
		rw.callBefore()
		// The status will be StatusOK if WriteHeader has not been called yet
		if rw.status == 0 {
			rw.status = http.StatusOK
		}

		if rw.hijacked {
			return
		}

		rw.ResponseWriter.WriteHeader(rw.status)
	})

	size, err := rw.ResponseWriter.Write(b)
	rw.size += size
	return size, err
}

func (rw *responseWriter) Status() int {
	return rw.status
}

func (rw *responseWriter) Size() int {
	return rw.size
}

func (rw *responseWriter) Written() bool {
	return rw.status != 0
}

func (rw *responseWriter) Before(before BeforeFunc) {
	rw.beforeFuncs = append(rw.beforeFuncs, before)
}

func (rw *responseWriter) Hijack() (rwc net.Conn, br *bufio.ReadWriter, err error) {
	hijacker, ok := rw.ResponseWriter.(http.Hijacker)
	if !ok {
		err = fmt.Errorf("the ResponseWriter doesn't support the Hijacker interface")
		return
	}
	rwc, br, err = hijacker.Hijack()
	if err != nil {
		return
	}
	rw.hijacked = true
	return
}

func (rw *responseWriter) CloseNotify() <-chan bool {
	return rw.ResponseWriter.(http.CloseNotifier).CloseNotify()
}

func (rw *responseWriter) Flush() {
	flusher, ok := rw.ResponseWriter.(http.Flusher)
	if ok {
		flusher.Flush()
	}
}

func (rw *responseWriter) callBefore() {
	for i := len(rw.beforeFuncs) - 1; i >= 0; i-- {
		rw.beforeFuncs[i](rw)
	}
}
