package servestk

import (
	"io"
	"io/ioutil"
	"net/http"
	"runtime/debug"

	"github.com/qiniu/http/httputil.v1"
	"github.com/qiniu/log.v1"
)

// ----------------------------------------------------------

var HandlePanic func(v interface{}) error

func SafeHandler(w http.ResponseWriter, req *http.Request, h func(w http.ResponseWriter, req *http.Request)) {
	defer func() {
		p := recover()
		if p != nil {
			log.Printf("WARN: panic fired in %v.panic - %v\n", h, p)
			log.Println(string(debug.Stack()))
			if HandlePanic != nil {
				e := HandlePanic(p)
				httputil.Error(w, e)
				return
			}
			w.WriteHeader(597)
		}
	}()
	h(w, req)
}

// ----------------------------------------------------------

func DiscardHandler(w http.ResponseWriter, req *http.Request, f func(w http.ResponseWriter, req *http.Request)) {
	f(w, req)
	io.Copy(ioutil.Discard, req.Body)
}

// ----------------------------------------------------------

type Mux interface {
	HandleFunc(pattern string, handler func(w http.ResponseWriter, req *http.Request))
	ServeHTTP(w http.ResponseWriter, req *http.Request)
}

type ServeStack struct {
	stk []func(http.ResponseWriter, *http.Request, func(w http.ResponseWriter, req *http.Request))
	Mux
}

func New(mux Mux, f ...func(http.ResponseWriter, *http.Request, func(w http.ResponseWriter, req *http.Request))) *ServeStack {
	if mux == nil {
		mux = http.DefaultServeMux
	}
	return &ServeStack{f, mux}
}

func (p *ServeStack) Push(f ...func(http.ResponseWriter, *http.Request, func(w http.ResponseWriter, req *http.Request))) {
	p.stk = append(p.stk, f...)
}

func (p *ServeStack) Build(h func(w http.ResponseWriter, req *http.Request)) func(http.ResponseWriter, *http.Request) {
	return BuildFunc(h, p.stk...)
}

func (p *ServeStack) HandleFunc(pattern string, handler func(http.ResponseWriter, *http.Request)) {
	p.Mux.HandleFunc(pattern, BuildFunc(handler, p.stk...))
}

func (p *ServeStack) Handle(pattern string, handler http.Handler) {
	p.Mux.HandleFunc(pattern, BuildFunc(handler.ServeHTTP, p.stk...))
}

type setDefaulter interface {
	SetDefault(h http.Handler)
}

// 若构造 ServeStack 时传入的 Mux 本身不支持 SetDefault，则调用此函数无意义
func (p *ServeStack) SetDefault(handler http.Handler) {
	if v, ok := p.Mux.(setDefaulter); ok {
		v.SetDefault(handler)
	}
}

// ----------------------------------------------------------

func BuildFunc(
	h func(http.ResponseWriter, *http.Request),
	stk ...func(http.ResponseWriter, *http.Request,
		func(w http.ResponseWriter, req *http.Request))) func(http.ResponseWriter, *http.Request) {

	if len(stk) == 0 {
		return h
	}
	return BuildFunc(func(w http.ResponseWriter, req *http.Request) {
		stk[0](w, req, h)
	}, stk[1:]...)
}

func Build(
	h http.Handler,
	stk ...func(http.ResponseWriter, *http.Request,
		func(w http.ResponseWriter, req *http.Request))) http.Handler {

	return http.HandlerFunc(BuildFunc(h.ServeHTTP, stk...))
}

// ----------------------------------------------------------
