package uri

import (
	"context"
	"io"
	"net/http"
	"net/url"

	"github.com/pkg/errors"
)

var ErrNotSupported = errors.New("not supported")
var ErrBadUri = errors.New("bad uri")

type Request struct {
	URI string

	beginOff *int64
	endOff   *int64
}

type GetOption func(*Request)

// Response 返回结果
type Response struct {
	Body   io.ReadCloser
	Header http.Header
	Size   int64
}

type Handler interface {
	Get(ctx context.Context, args Request, opts ...GetOption) (resp *Response, err error)
	Names() []string
}

func New(hs ...Handler) Handler {
	return newHandler(hs...)
}

type handler struct {
	h map[string]Handler // readonly
}

func newHandler(hs ...Handler) *handler {
	h2 := make(map[string]Handler)
	for _, h := range hs {
		for _, name := range h.Names() {
			if _, ok := h2[name]; ok {
				panic("durability schema:" + name)
			}
			h2[name] = h
		}

	}
	return &handler{h: h2}
}

func (h *handler) Get(ctx context.Context, args Request, opts ...GetOption,
) (resp *Response, err error) {

	u, err := url.Parse(args.URI)
	if err != nil {
		return nil, err
	}
	schema := u.Scheme
	handler, ok := h.h[schema]
	if ok {
		return handler.Get(ctx, args, opts...)
	}
	return nil, ErrNotSupported
}

func (h *handler) Names() (r []string) {
	for key := range h.h {
		r = append(r, key)
	}
	return
}
