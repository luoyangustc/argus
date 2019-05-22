package plugin

import (
	"reflect"

	"qiniupkg.com/httptest.v1/exec"
)

// ---------------------------------------------------------------------------

type subContext struct {
	ctx exec.IContext
}

func init() {

	exec.ExternalSub = new(subContext)
}

func (p *subContext) FindCmd(ctx exec.IContext, cmd string) reflect.Value {

	p.ctx = ctx
	v := reflect.ValueOf(p)
	return v.MethodByName("Eval_" + cmd)
}

// ---------------------------------------------------------------------------

