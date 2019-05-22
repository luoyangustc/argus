package httptest

import (
	"qiniupkg.com/httptest.v1"
	"qiniupkg.com/httptest.v1/exec"

	_ "qiniupkg.com/qiniutest/httptest.v1/exec/plugin"
)

// ---------------------------------------------------------------------------

type Context struct {
	*httptest.Context
	Ectx *exec.Context
}

func New(t httptest.TestingT) Context {

	ctx := httptest.New(t)
	ectx := exec.New()
	return Context{ctx, ectx}
}

func (p Context) Exec(code string) Context {

	p.Context.Exec(p.Ectx, code)
	return p
}

// ---------------------------------------------------------------------------

