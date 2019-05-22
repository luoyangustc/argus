package teapot

import (
	"fmt"
	"net/http"
	"net/url"
	"strings"
)

type Handler interface{}

type Handlers []Handler

func (h *Handlers) append(handlers ...Handler) {
	*h = append(*h, handlers...)
}

type nameParam string

func Name(name string) Handler {
	return nameParam(name)
}

type routeParams Handlers

func Router(path string, handlers ...Handler) Handler {
	path = strings.TrimSpace(path)
	if path == "" || path == "/" {
		panic("path can not be `" + path + "`")
	}
	return append(routeParams{path}, handlers...)
}

type RouteInfo struct {
	url.Values
	Path string
}

type paramList []map[string]string

func (d paramList) values() url.Values {
	values := make(url.Values)
	for _, m := range d {
		for key, value := range m {
			values.Set(key, value)
		}
	}
	return values
}

type pathParam struct {
	path string

	isParam   bool
	isWild    bool
	paramName string
}

func (d *pathParam) set(p string) {
	d.path = p

	if p[0] == ':' {
		d.isParam = true
		d.paramName = p[1:]
	}

	if p[0] == '*' && len(p) > 1 && p[1] == ':' {
		d.isWild = true
		d.paramName = p[2:]
	}
}

type routeRoot struct {
	*route

	notFoundFilters filters

	namedRoutes map[string]*route
}

func newRouteRoot() *routeRoot {
	routeRoot := new(routeRoot)
	routeRoot.route = newRoute(routeRoot, nil)
	routeRoot.isEnd = true
	return routeRoot
}

func (r *routeRoot) notFound(handlers ...interface{}) {
	r.notFoundFilters = makeFilters(handlers)
}

func (r *routeRoot) handle(ctx Context, rw http.ResponseWriter, req *http.Request) {
	// https://tools.ietf.org/html/rfc7231#section-4.1
	// By convention, standardized methods are defined in all-uppercase US-ASCII letters.
	req.Method = strings.ToUpper(req.Method)

	var (
		method = method(req.Method)
		params = make(paramList, 0)
		paths  = splitRoutePath(req.URL.Path)

		// root route
		route = r.route
	)

	if paths[0] != "" {
		// non root route go deep match
		route = r.match(paths, &params)
		if route != nil && !route.isEnd {
			route = nil
		}
	}

	// deal with not found
	if route == nil {
		ctx := newNestContext(ctx, rw.(ResponseWriter), r.notFoundFilters, nil)
		ctx.run()
		return
	}

	routeAction := route.action[method]
	actionFunc := method.action()

	if routeAction != nil {
		actionFunc = routeAction.action
	}

	if routeAction == nil && route.allRoute != nil {
		allRoute := route.allRoute
		if allRoute.controller.isFunc() ||
			allRoute.controller.actionFuncExists(actionFunc) {
			routeAction = allRoute

		} else if allRoute.action != "" {
			routeAction = allRoute
			actionFunc = allRoute.action
		}
	}

	if routeAction == nil && route.anyRoute != nil {
		anyRoute := route.anyRoute
		if anyRoute.controller.isFunc() ||
			anyRoute.controller.actionFuncExists(anyRoute.action) {
			routeAction = anyRoute
			actionFunc = anyRoute.action
		}
	}

	// deal with not found
	if routeAction == nil {
		ctx := newNestContext(ctx, rw.(ResponseWriter), r.notFoundFilters, nil)
		ctx.run()
		return
	}

	info := &RouteInfo{
		Values: params.values(),
		Path:   route.calcPath(),
	}
	ctx.Provide(info)

	// handle target route action
	nestCtx := newNestContext(ctx, rw.(ResponseWriter),
		routeAction.filters,
		routeAction.wrapHandle(route, params, actionFunc))
	nestCtx.run()
}

type route struct {
	pathParam pathParam
	isEnd     bool

	action map[method]*routerAction

	pathRoutes map[string]*route

	paramRoute *route
	wildRoute  *route

	root   *routeRoot
	parent *route

	allRoute *routerAction
	anyRoute *routerAction
}

func newRoute(routeRoot *routeRoot, parent *route) *route {
	return &route{
		root:       routeRoot,
		parent:     parent,
		action:     make(map[method]*routerAction),
		pathRoutes: make(map[string]*route),
	}
}

func (r *route) match(nextPaths []string, params *paramList) (rt *route) {
	p := nextPaths[0]
	paths := nextPaths[1:]
	more := len(paths) > 0

	// first match pathRoutes
	if rt = r.pathRoutes[p]; rt != nil {
		if more {
			rt = rt.match(paths, params)
		}
	}

	// second match paramRoute
	if rt == nil && r.paramRoute != nil {
		rt = r.paramRoute

		*params = append(*params, map[string]string{rt.pathParam.paramName: p})

		if more {
			rt = rt.match(paths, params)
		}
	}

	if rt != nil && rt.isEnd || r.wildRoute == nil {
		return
	}

	if rt = r.wildRoute; rt != nil {
		*params = append(*params, map[string]string{
			rt.pathParam.paramName: strings.Join(nextPaths, "/"),
		})
	}
	return
}

// config current router
func (r *route) configRoutes(args routeArgs) {
	args.filters = args.filters.remove(args.exempts...)

	var (
		allMethod methodParams
		allArgs   routeArgs
		anyMethod methodParams
		anyArgs   routeArgs
	)

	for _, mt := range args.methods {
		mArgs := calcRouterArgs(Handlers(mt))
		method := mArgs.method

		// detect All method
		if method.isAll() {
			allMethod = mt
			allArgs = mArgs
			continue
		}

		// detect Any method
		if method.isAny() {
			anyMethod = mt
			anyArgs = mArgs
			continue
		}

		act := string(mArgs.action)
		if act == "" {
			act = method.action()
		}

		actFilters := args.filters.append(mArgs.filters...).remove(mArgs.exempts...)
		r.action[method] = newRouteAction(mArgs.ctroler, act, actFilters)
	}

	// default use GET route for HEAD method
	if r.action[HEAD] == nil && r.action[GET] != nil {
		r.action[HEAD] = r.action[GET]
	}

	if allMethod != nil {
		act := string(allArgs.action)
		if act == "" && !allArgs.ctroler.isFunc() {
			if allArgs.ctroler.actionFuncExists("All") {
				act = "All"
			}
		}
		if act != "" {
			allArgs.ctroler.ensureHasAction(act)
		}
		actFilters := args.filters.append(allArgs.filters...).remove(allArgs.exempts...)
		r.allRoute = newRouteAction(allArgs.ctroler, act, actFilters)
	}

	if anyMethod != nil {
		act := string(anyArgs.action)
		if act == "" && !anyArgs.ctroler.isFunc() {
			act = "Any"
		}
		if act != "" {
			anyArgs.ctroler.ensureHasAction(act)
		}
		actFilters := args.filters.append(anyArgs.filters...).remove(anyArgs.exempts...)
		r.anyRoute = newRouteAction(anyArgs.ctroler, act, actFilters)
	}

	// loop nested routers
	for _, rtParam := range args.routers {
		rtArgs := calcRouterArgs(Handlers(rtParam))
		pathStr := rtArgs.strings[0]
		paths := splitRoutePath(pathStr)

		if paths[0] == "" {
			continue
		}

		targetRoute := r
		for i, p := range paths {
			var rt *route

			switch {
			case targetRoute.paramRoute != nil &&
				targetRoute.paramRoute.pathParam.path == p:
				rt = targetRoute.paramRoute

			case targetRoute.pathRoutes != nil &&
				targetRoute.pathRoutes[p] != nil:
				rt = targetRoute.pathRoutes[p]

			default:
				rt = newRoute(targetRoute.root, targetRoute)
				rt.pathParam.set(p)
			}

			if rt.pathParam.isParam {
				if targetRoute.paramRoute != nil {
					exists := targetRoute.paramRoute.pathParam.paramName
					setting := rt.pathParam.paramName
					if exists != setting {
						panic(fmt.Sprintf("route param conflict, please change `:%s` to `:%s`", setting, exists))
					}
				}

				targetRoute.paramRoute = rt
			} else if rt.pathParam.isWild {
				targetRoute.wildRoute = rt
			} else {
				targetRoute.pathRoutes[rt.pathParam.path] = rt
			}

			if i == len(paths)-1 {
				rt.isEnd = true
			}

			if rt.pathParam.isWild && !rt.isEnd {
				panic(fmt.Sprintf("wild route %s must on end of the route path %s",
					rt.pathParam.paramName, pathStr))
			}

			targetRoute = rt
		}

		rtArgs.filters = args.filters.append(rtArgs.filters...)

		targetRoute.configRoutes(rtArgs)
	}
}

func (r *route) calcPath() string {
	var path string
	rt := r
	for rt != nil {
		if rt.pathParam.path != "" {
			path = "/" + rt.pathParam.path + path
		}
		rt = rt.parent
	}
	if path == "" {
		path = "/"
	}
	return path
}
