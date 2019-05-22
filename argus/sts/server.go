package sts

import (
	"context"
	"io"
	"net/http"
	"strconv"
	"sync"
	"time"

	"github.com/qiniu/http/httputil.v1"
	"github.com/qiniu/http/restrpc.v1"
	"github.com/qiniu/xlog.v1"
	URIC "qiniu.com/argus/com/uri"
	URI "qiniu.com/argus/sts/uri"
)

type proxyArgs struct {
	URI    URI.Uri `json:"uri"`
	Length *int64  `json:"length"`
}

type fetchArgs struct {
	URI    URI.Uri `json:"uri"`
	Length *int64  `json:"length"`
	Sync   bool    `json:"sync"`
}

type fileArgs struct {
	CmdArgs []string
	ReqBody io.ReadCloser
	Length  *int64 `json:"length"`
}

type openArgs struct {
	CmdArgs []string
	Length  *int64 `json:"length"`
}

type writeArgs struct {
	CmdArgs []string
	ReqBody io.ReadCloser
}

// Server ...
type Server interface {
	// GET /proxy?uri=xxx&length=xxx
	GetProxy(ctx context.Context, args *proxyArgs, env *restrpc.Env)
	// POST /fetch?uri=xxx&length=xxx
	PostFetch(ctx context.Context, args *fetchArgs, env *restrpc.Env) error
	// GET /fetch?uri=xxx&length=xxx
	GetFetch(ctx context.Context, args *fetchArgs, env *restrpc.Env)
	// GET /file/xxx?length=xxx
	GetFile_(ctx context.Context, args *fileArgs, env *restrpc.Env)
	// POST /file/xxx?length=xxx[&open=true]
	PostFile_(ctx context.Context, uri *fileArgs, env *restrpc.Env) error
	// POST /open/xxx
	PostOpen_(ctx context.Context, uri *openArgs, env *restrpc.Env) (interface{}, error)
	// POST /write/xxx
	PostWrite_(ctx context.Context, args *writeArgs, env *restrpc.Env) error
}

type server struct {
	Fetcher
	Storage
	ws map[string]io.WriteCloser
	*sync.Mutex
}

// NewServer ...
func NewServer(storage Storage, fetcher Fetcher) Server {
	return &server{
		Fetcher: fetcher,
		Storage: storage,
		ws:      make(map[string]io.WriteCloser),
		Mutex:   new(sync.Mutex),
	}
}

func (s *server) initContext(ctx context.Context, env *restrpc.Env) (*xlog.Logger, context.Context) {
	xl, ok := xlog.FromContext(ctx)
	if !ok {
		xl = xlog.New(env.W, env.Req)
		ctx = xlog.NewContext(ctx, xl)
	}
	return xl, ctx
}

func (s *server) GetProxy(ctx context.Context, args *proxyArgs, env *restrpc.Env) {
	xl, ctx := s.initContext(ctx, env)
	xl.Infof("get proxy. %#v", args)

	if URI.TypeOf(args.URI) == URI.NONE {
		xl.Warnf("no support uri.")
		httputil.ReplyErr(env.W, ErrURINotSupport.Code, ErrURINotSupport.Err)
		return
	}

	var err error
	defer func(begin time.Time) {
		d := time.Since(begin)
		xl.Infof("get proxy done. %s %v", d, err)
		responseTime().
			WithLabelValues("server.GetProxy", formatError(err)).
			Observe(float64(d) / 1e9)
	}(time.Now())

	var opts = []URIC.GetOption{}
	if opt := URIC.ParseRangeRequest(ctx, env.Req.Header); opt != nil {
		opts = append(opts, opt)
	}

	resp, err := s.Fetcher.Get(ctx, args.URI, opts...)
	if err != nil {
		xl.Errorf("get proxy failed. %s %s", args.URI, err)
		code, desc := httputil.DetectError(err)
		httputil.ReplyErr(env.W, code, desc)
		err = httputil.NewError(code, desc)
		return
	}
	defer resp.Body.Close()

	env.W.Header().Set("Content-Length", strconv.FormatInt(resp.Size, 10))
	for key, value := range resp.Header {
		env.W.Header()[key] = value
	}
	env.W.WriteHeader(http.StatusOK) // 先写Header，再填Code

	_, err = io.Copy(env.W, resp.Body)
	if err != nil {
		xl.Warnf("copy to response body failed. %v", err)
	}
	return
}

func (s *server) PostFetch(ctx context.Context, args *fetchArgs, env *restrpc.Env) error {
	xl, ctx := s.initContext(ctx, env)
	xl.Infof("post fetch. %#v", args)

	if URI.TypeOf(args.URI) == URI.NONE {
		xl.Warnf("no support uri.")
		return ErrURINotSupport
	}

	var err error
	defer func(begin time.Time) {
		d := time.Since(begin)
		xl.Infof("post fetch done. %s %v", d, err)
		responseTime().
			WithLabelValues("server.PostFetch", formatError(err)).
			Observe(float64(d) / 1e9)
	}(time.Now())

	err = s.Storage.Fetch(ctx, args.URI, args.Length, args.Sync)
	return err
}

func (s *server) GetFetch(ctx context.Context, args *fetchArgs, env *restrpc.Env) {
	xl, ctx := s.initContext(ctx, env)
	xl.Infof("get fetch. %#v", args)

	if URI.TypeOf(args.URI) == URI.NONE {
		xl.Warnf("no support uri.")
		httputil.ReplyErr(env.W, ErrURINotSupport.Code, ErrURINotSupport.Err)
		return
	}

	var err error
	defer func(begin time.Time) {
		d := time.Since(begin)
		xl.Infof("get fetch done. %s %v", d, err)
		responseTime().
			WithLabelValues("server.GetFetch", formatError(err)).
			Observe(float64(d) / 1e9)
	}(time.Now())

	reader, length, err := s.Storage.Get(ctx, args.URI, args.Length, false)
	if err != nil {
		xl.Errorf("get fetch failed. %s %s", args.URI, err)
		code, desc := httputil.DetectError(err)
		httputil.ReplyErr(env.W, code, desc)
		err = httputil.NewError(code, desc)
		return
	}
	defer reader.Close()

	env.W.Header().Set("Content-Length", strconv.FormatInt(length, 10))
	env.W.WriteHeader(http.StatusOK)

	_, err = io.Copy(env.W, reader)
	if err != nil {
		xl.Warnf("copy to response body failed. %v", err)
	}
	return
}

func (s *server) GetFile_(ctx context.Context, args *fileArgs, env *restrpc.Env) {
	xl, ctx := s.initContext(ctx, env)
	xl.Infof("get file. %#v", args)

	var err error
	defer func(begin time.Time) {
		d := time.Since(begin)
		xl.Infof("get file done. %s %v", d, err)
		responseTime().
			WithLabelValues("server.GetFile", formatError(err)).
			Observe(float64(d) / 1e9)
	}(time.Now())

	uri := URI.NewURI("sts:///" + args.CmdArgs[0])
	reader, length, err := s.Storage.Get(ctx, uri, args.Length, false)
	if err != nil {
		xl.Errorf("get failed. %d %s", args.Length, err)
		code, desc := httputil.DetectError(err)
		httputil.ReplyErr(env.W, code, desc)
		err = httputil.NewError(code, desc)
		return
	}
	defer reader.Close()

	env.W.Header().Set("Content-Length", strconv.FormatInt(length, 10))
	env.W.WriteHeader(http.StatusOK)

	_, err = io.Copy(env.W, reader)
	if err != nil {
		xl.Warnf("copy to response body failed. %v", err)
	}
	return
}

func (s *server) PostFile_(ctx context.Context, req *fileArgs, env *restrpc.Env) error {
	xl, ctx := s.initContext(ctx, env)
	xl.Infof("post file. %s %#v", env.Req.URL.String(), req)

	var err error
	defer func(begin time.Time) {
		d := time.Since(begin)
		xl.Infof("post file done. %s %v", d, err)
		responseTime().
			WithLabelValues("server.PostFile", formatError(err)).
			Observe(float64(d) / 1e9)
	}(time.Now())

	uri := URI.NewURI("sts:///" + req.CmdArgs[0])
	err = s.Post(ctx, uri, env.Req.ContentLength, req.ReqBody)
	if err == nil {
		return nil
	}
	xl.Errorf("post file failed. %d %v", env.Req.ContentLength, err)
	return err
}

func (s *server) PostOpen_(ctx context.Context, req *openArgs, env *restrpc.Env) (interface{}, error) {
	xl, ctx := s.initContext(ctx, env)
	xl.Infof("open file. %s %#v", env.Req.URL.String(), req)

	var err error
	defer func(begin time.Time) {
		d := time.Since(begin)
		xl.Infof("post open done. %s %v", d, err)
		responseTime().
			WithLabelValues("server.PostOpen", formatError(err)).
			Observe(float64(d) / 1e9)
	}(time.Now())

	uri := URI.NewURI("sts:///" + req.CmdArgs[0])
	writer, err := s.Open(ctx, uri, *req.Length)
	if err != nil {
		return nil, err
	}
	id := xlog.GenReqId()
	s.Lock()
	defer s.Unlock()
	s.ws[id] = writer
	go func(xl *xlog.Logger) {
		time.AfterFunc(time.Second*5,
			func() {
				s.Lock()
				defer s.Unlock()
				if w, ok := s.ws[id]; ok {
					delete(s.ws, id)
					w.Close()
					xl.Infof("writer overdue. %s", uri.ToString())
					_OverdueOpens.Inc()
				}
			},
		)
	}(xl.Spawn())
	return struct {
		ID string `json:"id"`
	}{ID: id}, nil
}

func (s *server) PostWrite_(ctx context.Context, req *writeArgs, env *restrpc.Env) error {
	xl, ctx := s.initContext(ctx, env)
	xl.Infof("write file. %s %#v", env.Req.URL.String(), req)

	var err error
	defer func(begin time.Time) {
		d := time.Since(begin)
		xl.Infof("post write done. %s %v", d, err)
		responseTime().
			WithLabelValues("server.PostWrite", formatError(err)).
			Observe(float64(d) / 1e9)
	}(time.Now())

	s.Lock()
	writer, ok := s.ws[req.CmdArgs[0]]
	if !ok {
		s.Unlock()
		xl.Warnf("write file failed. %s overdue", req.CmdArgs[0])
		return ErrOverdue
	}
	delete(s.ws, req.CmdArgs[0])
	s.Unlock()

	defer writer.Close()
	_, err = io.Copy(writer, req.ReqBody)
	if err == nil {
		return nil
	}
	xl.Errorf("post file failed. %d %v", env.Req.ContentLength, err)
	err = ErrIO
	return err
}
