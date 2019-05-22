package teapot

import (
	"io"
	"net/http"
	"reflect"
)

type ActionOut interface {
	Out() []reflect.Value
}

type actionOut []reflect.Value

func (r actionOut) Out() []reflect.Value {
	return r
}

type ActionResult interface {
	Write(ctx Context, rw http.ResponseWriter, req *http.Request)
}

// GenericOutFilter is a filter use parse generic response result
// eg:
// Action() (string, int)
// Action() ([]byte, int)
// Action() (io.Reader, int)
// Action() string
// Action() []byte
// Action() io.Reader
// Action() int
// Action() (Result)
func GenericOutFilter() interface{} {
	return func(ctx Context, rw http.ResponseWriter, req *http.Request) {
		ctx.Next()

		if ctx.Written() {
			return
		}

		var res ActionOut
		ctx.Find(&res, "")
		if res == nil {
			return
		}

		out := res.Out()
		if len(out) == 0 {
			return
		}

		var body reflect.Value
		if out[len(out)-1].Kind() == reflect.Int {
			code := out[len(out)-1].Int()
			rw.WriteHeader(int(code))

			if len(out) == 1 {
				return
			}

			body = out[len(out)-2]
		} else {
			body = out[len(out)-1]
		}

		if !body.CanInterface() {
			return
		}

		itf := body.Interface()
		switch src := itf.(type) {
		case string:
			rw.Write([]byte(src))
		case []byte:
			rw.Write(src)
		case io.Reader:
			io.Copy(rw, src)
		case ActionResult:
			src.Write(ctx, rw, req)
		case http.Handler:
			src.ServeHTTP(rw, req)
		}
	}
}
