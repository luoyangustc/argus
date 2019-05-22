package scenario

import (
	"flag"
	"fmt"
	"os"
	"os/signal"
	"syscall"

	"github.com/qiniu/xlog.v1"
)

type Scenario interface {
	Prepare() error
	Init() error
	Info() error
	Run() error
}

func Main(s Scenario, register func() error) {

	errs := make(chan error)
	go func() {
		c := make(chan os.Signal, 1)
		signal.Notify(c, syscall.SIGINT, syscall.SIGTERM)
		errs <- fmt.Errorf("%s", <-c)
	}()

	var (
		xl         = xlog.NewWith("MAIN")
		err        error
		isInfoMode bool
	)

	flag.BoolVar(&isInfoMode, "info", false, "info cmd")

	if err = s.Prepare(); err != nil {
		xl.Fatalf("prepare failed. err: %v", err)
	}

	if err = register(); err != nil {
		xl.Fatalf("register failed. err: %v", err)
	}

	if err = s.Init(); err != nil {
		xl.Fatalf("init failed, err: %v", err)
	}

	if !flag.Parsed() {
		flag.Parse()
	}
	if isInfoMode {
		err = s.Info()
		if err != nil {
			xl.Fatalf("dump app info failed, err: %v", err)
		}
		return
	}

	go func() {
		errs <- s.Run()
	}()

	xl.Fatalf("err: %v", <-errs)
}
