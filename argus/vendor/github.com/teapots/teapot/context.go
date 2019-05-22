package teapot

import (
	"github.com/teapots/inject"
)

type Context interface {
	// per request Injector
	inject.Injector

	// Next is an optional function that Middleware Handlers can call to yield the until after
	// the other Handlers have been executed. This works really well for any operations that must
	// happen after an http request
	Next()

	// Written returns whether or not the response for this context has been change.
	Written() bool
}

type context struct {
	inject.Injector

	filters filters

	action FilterFunc

	rw    ResponseWriter
	index int
}

var _ Context = new(context)

func newContext(rw ResponseWriter, filters []*filter, actionFunc interface{}) *context {
	ctx := &context{
		Injector: inject.New(),

		rw:      rw,
		filters: filters,
	}

	ctx.Injector.ProvideAs(ctx, (*Context)(nil))

	if actionFunc != nil {
		ctx.action = func() inject.Provider { return actionFunc }
	}

	return ctx
}

func (c *context) Next() {
	c.index += 1
	c.run()
}

func (c *context) Written() bool {
	return c.rw.Written()
}

func (c *context) handler() FilterFunc {
	if c.index < len(c.filters) {
		return c.filters[c.index].fun
	}
	if c.index == len(c.filters) {
		return c.action
	}
	panic("invalid index for context handler")
}

func (c *context) run() {
	for c.index <= len(c.filters) {
		f := c.handler()
		if f == nil {
			c.index += 1
			continue
		}

		_, err := c.Invoke(f())
		if err != nil {
			panic(err)
		}
		c.index += 1

		if c.Written() {
			return
		}
	}
}

type nestContext struct {
	context
}

func newNestContext(inj inject.Injector, rw ResponseWriter, filters []*filter, actionFunc interface{}) *nestContext {
	ctx := &nestContext{
		context{
			Injector: inj,

			rw:      rw,
			filters: filters,
		},
	}

	inj.ProvideAs(ctx, (*Context)(nil))

	if actionFunc != nil {
		ctx.action = func() inject.Provider { return actionFunc }
	}
	return ctx
}
