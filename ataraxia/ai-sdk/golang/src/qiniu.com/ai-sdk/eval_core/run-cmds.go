package main

import (
	"context"
	"os"
	"os/signal"
	"syscall"
	"time"
)

func runCmds() {
	ctx, cancel := context.WithCancel(context.Background())
	ch := make(chan string, len(cfg.Process))
	for _, process := range cfg.Process {
		go func(process Process) {
			var env []string
			if process.WithSysEnv {
				env = os.Environ()
				env = append(env, process.Env...)
			} else {
				env = append(process.Env)
			}
			runCmd(ctx, process.Name, process.Dir, process.Cmd, process.Args, env)
			ch <- "exit"
		}(process)
	}
	clean := func() {
		xl.Info("cancel()")
		cancel()
		time.Sleep(time.Second)
		xl.Info("os.Exit(1)")
		os.Exit(1)
	}
	c := make(chan os.Signal, 1)
	signal.Notify(c, os.Interrupt, syscall.SIGTERM)
	select {
	case <-c:
		clean()
	case <-ch:
		clean()
	}
}
