package teapot

const (
	GET     method = "GET"
	POST    method = "POST"
	PUT     method = "PUT"
	HEAD    method = "HEAD"
	DELETE  method = "DELETE"
	OPTIONS method = "OPTIONS"
	PATCH   method = "PATCH"
	TRACE   method = "TRACE"
	CONNECT method = "CONNECT"
)

var (
	methods = []method{GET, POST, PUT, HEAD, DELETE, OPTIONS, PATCH, TRACE, CONNECT}
)

type method string

func (m method) isAll() bool {
	return m == "*"
}

func (m method) isAny() bool {
	return m == "**"
}

func (m method) action() string {
	chars := make([]rune, 0, len(m))
	for i, s := range m {
		if i != 0 {
			s += 32
		}
		chars = append(chars, s)
	}
	return string(chars)
}

type MethodHandler interface {
	Action(string) MethodHandler
	Filter(...interface{}) MethodHandler
	Exempt(...interface{}) MethodHandler
}

type methodParams Handlers

func (h methodParams) Action(name string) MethodHandler {
	// ensure controller has action
	rc := h[0].(routerController)
	rc.ensureHasAction(name)
	h[0] = rc

	h = append(h, action(name))
	return h
}

func (h methodParams) Filter(exempts ...interface{}) MethodHandler {
	h = append(h, Filter(exempts...))
	return h
}

func (h methodParams) Exempt(exempts ...interface{}) MethodHandler {
	h = append(h, Exempt(exempts...))
	return h
}

func Get(c Handler) MethodHandler {
	return methodParams{makeRouterController(c), GET}
}

func Post(c Handler) MethodHandler {
	return methodParams{makeRouterController(c), POST}
}

func Put(c Handler) MethodHandler {
	return methodParams{makeRouterController(c), PUT}
}

func Head(c Handler) MethodHandler {
	return methodParams{makeRouterController(c), HEAD}
}

func Delete(c Handler) MethodHandler {
	return methodParams{makeRouterController(c), DELETE}
}

func Options(c Handler) MethodHandler {
	return methodParams{makeRouterController(c), OPTIONS}
}

func Patch(c Handler) MethodHandler {
	return methodParams{makeRouterController(c), PATCH}
}

func Trace(c Handler) MethodHandler {
	return methodParams{makeRouterController(c), TRACE}
}

func Connect(c Handler) MethodHandler {
	return methodParams{makeRouterController(c), CONNECT}
}

func All(c Handler) MethodHandler {
	return methodParams{makeRouterController(c), method("*")}
}

func Any(c Handler) MethodHandler {
	return methodParams{makeRouterController(c), method("**")}
}
