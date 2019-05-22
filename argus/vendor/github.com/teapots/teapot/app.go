package teapot

import (
	"fmt"
	"log"
	"net/http"
	"os"

	"github.com/teapots/inject"
)

const HeaderPoweredBy = "X-Powered-By"

type Teapot struct {
	route *routeRoot

	// global filters
	filters filters

	// teapot app logger
	logger LoggerAdv

	// master inject of current app
	inject inject.Injector

	// app server
	Server *http.Server

	// app config
	Config *Config
}

var _ inject.TypeProvider = new(Teapot)

func New() *Teapot {
	teapot := &Teapot{
		route:  newRouteRoot(),
		inject: inject.New(),
		Config: newConfig(),
	}

	teapot.Server = &http.Server{
		Handler: teapot,
	}

	teapot.Provide(teapot.Config)

	log := NewLogger(log.New(os.Stderr, "", log.LstdFlags|log.Lmicroseconds))
	log.SetColorMode(true)
	teapot.SetLogger(log)

	teapot.NotFound(defaultNotFound)
	return teapot
}

func (t *Teapot) ImportConfig(c Configer) {
	t.Config.setParent(c)
	t.Config.Bind(&t.Config.RunPath, "run_path")
	t.Config.Bind(&t.Config.RunMode, "run_mode")
	t.Config.Bind(&t.Config.HttpAddr, "http_addr")
	t.Config.Bind(&t.Config.HttpPort, "http_port")
}

func (t *Teapot) NotFound(handlers ...interface{}) {
	t.route.notFound(handlers...)
}

func (t *Teapot) Logger() Logger {
	return t.logger
}

func (t *Teapot) SetLogger(logger LoggerAdv) {
	t.logger = logger
	t.ProvideAs(logger, (*Logger)(nil))
}

func (t *Teapot) Injector() inject.Injector {
	return t.inject
}

func (t *Teapot) Provide(provs ...interface{}) inject.TypeProvider {
	return t.inject.Provide(provs...)
}

func (t *Teapot) ProvideAs(prov interface{}, typ interface{}) inject.TypeProvider {
	return t.inject.ProvideAs(prov, typ)
}

func (t *Teapot) Filter(handlers ...interface{}) {
	t.filters = t.filters.append(makeFilters(handlers)...)
}

func (t *Teapot) Routers(handlers ...Handler) *Teapot {
	args := calcRouterArgs(handlers)

	t.route.configRoutes(args)
	return t
}

func (t *Teapot) ServeHTTP(rw http.ResponseWriter, req *http.Request) {
	// server info
	rw.Header().Set(HeaderPoweredBy, "Teapot")

	// wrap http.ResponseWriter
	trw := newResponseWriter(rw)

	ctx := newContext(trw, t.filters, t.route.handle)
	ctx.SetParent(t.inject)
	ctx.Provide(req)
	ctx.ProvideAs(trw, (*http.ResponseWriter)(nil))
	ctx.run()

	// flush header if have not written
	trw.Write(nil)
}

func (t *Teapot) Run() error {
	mode := string(t.Config.RunMode)
	addr := fmt.Sprintf("%s:%s", t.Config.HttpAddr, t.Config.HttpPort)

	t.Server.Addr = addr

	if t.Config.RunMode.IsProd() {
		t.logger.SetColorMode(false)
	} else {
		addr = newBrush("32")(t.Server.Addr)
		mode = newBrush("32")(mode)
	}

	t.logger.Infof("Teapot listening on %s in [%s] mode", addr, mode)
	err := t.Server.ListenAndServe()
	t.logger.Emergency(err)
	return err
}
