package teapot

import (
	"net/http"
	"path"
	"strings"
)

type routeArgs struct {
	name    nameParam
	strings []string
	filters filters
	exempts filters
	routers []routeParams
	methods []methodParams
	ctroler *routerController
	method  method
	action  action
}

func calcRouterArgs(handlers Handlers) routeArgs {
	var args routeArgs
	for _, h := range handlers {
		switch arg := h.(type) {
		case string:
			args.strings = append(args.strings, arg)
		case includeFilter:
			args.filters = args.filters.append(arg...)
		case exemptFilter:
			args.exempts = args.exempts.append(arg...)
		case routeParams:
			args.routers = append(args.routers, arg)
		case methodParams:
			args.methods = append(args.methods, arg)
		case nameParam:
			args.name = arg
		case routerController:
			args.ctroler = &arg
		case action:
			args.action = arg
		case method:
			args.method = arg
		}
	}

	if args.ctroler != nil {
		// ensure controller has action

		if args.action == "" &&
			!args.ctroler.isFunc() &&
			!args.method.isAll() &&
			!args.method.isAny() {
			args.ctroler.ensureHasAction(args.method.action())
		}
	}

	return args
}

func splitRoutePath(p string) []string {
	// keep first value in []string is empty
	p = path.Clean("/" + strings.TrimSpace(p))
	splits := strings.Split(p, "/")

	// return non-empty parts
	return splits[1:]
}

func handleStatus(rw http.ResponseWriter, status int) {
	rw.WriteHeader(status)
	rw.Write([]byte(ToStr(status) + " " + http.StatusText(status)))
}
