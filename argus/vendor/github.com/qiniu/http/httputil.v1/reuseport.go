// +build linux darwin

package httputil

import (
	"errors"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/facebookgo/httpdown"
	"github.com/kavu/go_reuseport"
)

var (
	ErrReusePortNotSupport = errors.New("SO_REUSEPORT is not supported")
)

type ServeOption struct {
	StopFunc func() error

	ForceReusePort bool
	StopTimeout    time.Duration
	KillTimeout    time.Duration
}

func ListenAndServe(addr string, handler http.Handler, opt ServeOption) (err error) {

	if opt.StopFunc == nil {
		opt.StopFunc = func() error { return nil }
	}

	l, err := reuseport.NewReusablePortListener("tcp4", addr)
	if err != nil {
		if strings.Contains(err.Error(), "protocol not available") {
			log.Println("SO_REUSEPORT is not supported in this kernal")

			// force SO_REUSEPORT
			if opt.ForceReusePort {
				return ErrReusePortNotSupport
			}

			// listen without SO_REUSEPORT
			l, err = net.Listen("tcp", addr)
			if err != nil {
				return
			}
		} else {
			return
		}
	}
	defer l.Close()

	server := &http.Server{
		Handler: handler,
	}
	if opt.StopTimeout <= 0 {
		opt.StopTimeout = time.Second * 300
	}
	if opt.KillTimeout <= 0 {
		opt.KillTimeout = time.Second * 60
	}

	hd := &httpdown.HTTP{
		StopTimeout: opt.StopTimeout,
		KillTimeout: opt.KillTimeout,
	}

	service := hd.Serve(server, l)

	waitForInterrupt(func() {
		if err = service.Stop(); err != nil {
			log.Println("stop http server failed:", err)
		}
		if err = opt.StopFunc(); err != nil {
			log.Println("stop service failed:", err)
		}
	})
	log.Println("stopped")

	return
}

func waitForInterrupt(interrupt func()) {

	c := make(chan os.Signal, 1)
	signal.Notify(c, syscall.SIGTERM, os.Interrupt, os.Kill)
	s := <-c
	log.Println("Receiving signal:", s)

	interrupt()

	signal.Stop(c)
}
