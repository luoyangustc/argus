package server

import (
	"context"
	"encoding/json"
	"sync"
	"time"
)

type (
	EvalConfig struct {
		Host      string `json:"host"`
		TimeoutMS int64  `json:"timeout_ms"`
		Timeout   time.Duration
	}

	Config struct {
		EvalDefault EvalConfig                 `json:"eval_default"`
		Evals       map[string]EvalConfig      `json:"evals"`
		Handlers    map[string]json.RawMessage `json:"handlers"`
	}

	NewEval  func(EvalConfig) interface{}
	IHandler interface {
		Init(json.RawMessage, IServer) interface{}
	}
	IServer interface {
		GetEval(string) interface{}
		IImageParse
	}

	Server struct {
		Config
		IImageParse

		evalF map[string]NewEval

		evals    map[string]interface{}
		handlers map[string]IHandler

		sync.Mutex
	}
)

func (c Config) getEval(name string) EvalConfig {
	var cfg EvalConfig
	if c.Evals != nil {
		cfg = c.Evals[name]
	}

	if cfg.TimeoutMS == 0 {
		cfg.TimeoutMS = c.EvalDefault.TimeoutMS
	}
	if cfg.Host == "" {
		cfg.Host = c.EvalDefault.Host
	}

	cfg.Timeout = time.Millisecond * time.Duration(cfg.TimeoutMS)
	if cfg.Timeout == 0 {
		cfg.Timeout = time.Second * 30
	}

	return cfg
}

func NewServer() *Server {
	return &Server{
		evalF:    make(map[string]NewEval),
		evals:    make(map[string]interface{}),
		handlers: make(map[string]IHandler),
		Mutex:    sync.Mutex{},
	}
}

func (s *Server) Init(c Config, ip IImageParse) *Server {
	s.Config = c
	s.IImageParse = ip
	return s
}

func (s *Server) RegisterEval(name string, newf NewEval) {
	s.evalF[name] = newf
}

func (s *Server) RegisterHandler(name string, handler IHandler) {
	s.handlers[name] = handler
}

func (s *Server) GetEval(name string) interface{} {
	s.Lock()
	defer s.Unlock()

	eval, ok := s.evals[name]
	if ok {
		return eval
	}
	newF, ok := s.evalF[name]
	if !ok {
		return nil
	}
	eval = newF(s.Config.getEval(name))
	if eval == nil {
		return nil
	}
	s.evals[name] = eval
	return eval
}

func (s *Server) ParseImage(ctx context.Context, uri string) (Image, error) {
	if s.IImageParse == nil {
		return Image{}, nil
	}
	return s.IImageParse.ParseImage(ctx, uri)
}

func (s *Server) Handlers() []interface{} {
	handlers := make([]interface{}, 0, len(s.handlers))
	for name, ih := range s.handlers {
		handlers = append(handlers, ih.Init(s.Config.Handlers[name], s))
	}
	return handlers
}

////////////////////////////////////////////////////////////////////////////////

var DefaultServer = NewServer()

func RegisterEval(name string, newf NewEval) {
	DefaultServer.RegisterEval(name, newf)
}

func RegisterHandler(name string, handler IHandler) {
	DefaultServer.RegisterHandler(name, handler)
}
