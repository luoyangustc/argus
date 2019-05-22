package server

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestConfig(t *testing.T) {
	var cfg Config
	err := json.Unmarshal(
		[]byte(`{
	"eval_default": {"host": "test.com", "timeout_ms": 300},
	"evals": {
		"a": {"host": "a.com", "timeout_ms": 100},
		"b": {"host": "b.com"},
		"c": {"timeout_ms": 200},
		"d": {}
	},
	"handlers": {}
		}`),
		&cfg,
	)
	assert.NoError(t, err)

	{
		c := cfg.getEval("e")
		assert.Equal(t, "test.com", c.Host)
		assert.Equal(t, time.Millisecond*300, c.Timeout)
	}
	{
		c := cfg.getEval("a")
		assert.Equal(t, "a.com", c.Host)
		assert.Equal(t, time.Millisecond*100, c.Timeout)
	}
	{
		c := cfg.getEval("b")
		assert.Equal(t, "b.com", c.Host)
		assert.Equal(t, time.Millisecond*300, c.Timeout)
	}
	{
		c := cfg.getEval("c")
		assert.Equal(t, "test.com", c.Host)
		assert.Equal(t, time.Millisecond*200, c.Timeout)
	}
	{
		c := cfg.getEval("d")
		assert.Equal(t, "test.com", c.Host)
		assert.Equal(t, time.Millisecond*300, c.Timeout)
	}
}

type mockHandler struct{}

func (mock mockHandler) Init(json.RawMessage, IServer) interface{} { return mock }

func TestServer(t *testing.T) {

	var cfg Config
	err := json.Unmarshal(
		[]byte(`{
	"eval_default": {"host": "test.com", "timeout_ms": 300},
	"evals": {},
	"handlers": {}
		}`),
		&cfg,
	)
	assert.NoError(t, err)

	var s = NewServer().Init(cfg, nil)

	assert.Nil(t, s.GetEval("a"))
	s.RegisterEval("b", func(EvalConfig) interface{} { return 10 })
	assert.Equal(t, 10, s.GetEval("b").(int))
	s.RegisterEval("c", func(EvalConfig) interface{} { return true })
	assert.Equal(t, true, s.GetEval("c").(bool))

	s.RegisterHandler("A", mockHandler{})
	s.RegisterHandler("B", mockHandler{})
	s.RegisterHandler("C", mockHandler{})
	handlers := s.Handlers()
	assert.Equal(t, 3, len(handlers))
}
