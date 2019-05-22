package teapot

import (
	"fmt"
	"net/http"
	"reflect"
)

type action string

type routerController struct {
	value interface{}
	val   reflect.Value
	typ   reflect.Type
	kind  reflect.Kind
	pTyp  reflect.Type

	// struct action
	actionIn  []reflect.Type
	actionOut []reflect.Type
}

func makeRouterController(value interface{}) (rc routerController) {
	val := reflect.ValueOf(value)

	iTyp := indirectType(val.Type())

	switch iTyp.Kind() {
	case reflect.Struct, reflect.Func:
	default:
		panic(fmt.Sprintf("action must be a struct or function but found `%v`", val.Kind()))
	}

	rc.value = value
	rc.val = val
	rc.typ = iTyp
	rc.kind = val.Kind()
	rc.pTyp = reflect.New(rc.typ).Type()
	return
}

func (r *routerController) isFunc() bool {
	return r.kind == reflect.Func
}

func (r *routerController) ensureHasAction(action string) {
	if r.typ.Kind() != reflect.Struct {
		if action != "" {
			panic("function controller don't need action method")
		}
		return
	}

	m := reflect.New(r.typ).MethodByName(action)
	if m.IsValid() {
		r.cacheStructAction(m.Type())
		return
	}

	panic(fmt.Sprintf("controller action `%s.%s` not exists", r.typ, action))
}

func (r *routerController) cacheStructAction(typ reflect.Type) {
	r.actionIn = make([]reflect.Type, typ.NumIn())
	for i := 0; i < typ.NumIn(); i++ {
		r.actionIn[i] = typ.In(i)
	}
	r.actionOut = make([]reflect.Type, typ.NumOut())
	for i := 0; i < typ.NumOut(); i++ {
		r.actionOut[i] = typ.Out(i)
	}
}

func (r *routerController) actionFuncExists(action string) bool {
	_, ok := r.pTyp.MethodByName(action)
	return ok
}

type routerAction struct {
	controller *routerController
	action     string
	filters    filters
}

func newRouteAction(c *routerController, a string, f []*filter) *routerAction {
	return &routerAction{
		controller: c,
		action:     a,
		filters:    f,
	}
}

func (r *routerAction) wrapHandle(route *route, params paramList, action string) interface{} {
	return func(ctx Context, rw http.ResponseWriter, req *http.Request) {

		var out []reflect.Value

		if r.controller.isFunc() {
			var err error
			// invoke function controller
			out, err = ctx.Invoke(r.controller.value)
			if err != nil {
				panic(err)
			}
		} else {
			out = r.callStructFunc(ctx, params, action)
		}

		r.writeResult(ctx, out)
	}
}

func (r *routerAction) callStructFunc(ctx Context, params paramList, action string) []reflect.Value {
	actionIn := r.controller.actionIn

	in := make([]reflect.Value, 0, len(actionIn))
	for i := 0; i < len(actionIn); i++ {
		typ := actionIn[i]
		value := reflect.Value{}

		if len(params) > i {
			param := ""
			for _, v := range params[i] {
				param = v
			}
			value = convertStringAsType(param, typ)
		}

		if !value.IsValid() {
			value = reflect.New(typ).Elem()
		}

		in = append(in, value)
	}

	if !r.controller.actionFuncExists(action) {
		panic(fmt.Sprintf("not found action func: %s", action))
	}

	newStruct := reflect.New(r.controller.typ)

	err := ctx.Apply(newStruct.Interface())
	if err != nil {
		panic(err)
	}

	out := newStruct.MethodByName(action).Call(in)
	return out
}

func (r *routerAction) writeResult(ctx Context, out []reflect.Value) {
	res := actionOut(out)
	ctx.ProvideAs(&res, (*ActionOut)(nil))
}
